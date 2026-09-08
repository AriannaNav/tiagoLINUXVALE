"""One LLM call per turn returns both the spoken reply and a structured action,
constrained by the JSON schema below.

Every decision — what to serve, whether to refuse, the customer's mood and urgency
— is the model's own. Python only loads the menu, builds the prompt, sanitises the
spoken text and passes the decisions through. Model: WAITER_AGENT_MODEL, default
qwen2.5:7b. A malformed answer degrades to a clarification request."""
import json
import os
import re

import requests

try:
    from config import ADULT_AGE_THRESHOLD as _ADULT_AGE
except Exception:
    _ADULT_AGE = 18

_EMOJI_RE = re.compile(
    "[\U0001F000-\U0001FAFF\U00002600-\U000027BF\U00002B00-\U00002BFF"
    "\U0001F1E6-\U0001F1FF\U00002190-\U000021FF️]")
_JUNK_RE = re.compile(
    r">{2,}|\[(?:serve|decline|clarify|bathroom|reception|smalltalk|end)[^\]]*\]",
    re.I)

def _sanitize_say(text):
    text = _JUNK_RE.sub(" ", str(text))
    text = _EMOJI_RE.sub("", text)
    text = re.sub(r"\s+", " ", text).strip().strip('"').strip()
    sents = re.split(r"(?<=[.!?])\s+", text)
    if len(sents) > 2:
        text = " ".join(sents[:2]).strip()
    return text

try:
    from config import OLLAMA_URL
except Exception:
    OLLAMA_URL = "http://localhost:11434/api/generate"

CHAT_URL = OLLAMA_URL.replace("/api/generate", "/api/chat")
AGENT_MODEL = os.environ.get("WAITER_AGENT_MODEL", "qwen2.5:7b")
AGENT_TEMP = float(os.environ.get("WAITER_AGENT_TEMP", "0.5"))
# CPU-only inference runs about half a token per second here, so a turn can
# take well over a minute. At 60s every reply timed out and the agent fell
# back to "clarify", which looked like the robot never understanding anything.
AGENT_TIMEOUT = float(os.environ.get("WAITER_AGENT_TIMEOUT", "240"))
AGENT_MAX_TOKENS = int(os.environ.get("WAITER_AGENT_MAX_TOKENS", "160"))

_MENU_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "menu.yaml")

_DEFAULT_ITEMS = [
    {"name": "Coca-Cola", "serve": "coca cola", "category": "drink",
     "synonyms": ["coke", "cola", "coca"]},
    {"name": "Sprite", "serve": "sprite", "category": "drink", "synonyms": []},
    {"name": "water", "serve": "water", "category": "drink", "synonyms": []},
    {"name": "orange juice", "serve": "juice", "category": "drink",
     "synonyms": ["juice", "orange"]},
    {"name": "wine", "serve": "wine", "category": "drink", "adults_only": True,
     "synonyms": []},
    {"name": "Pringles", "serve": "pringles", "category": "food",
     "synonyms": ["chips", "crisps"]},
]

def _load_menu(path=_MENU_FILE):
    try:
        import yaml
        with open(path) as f:
            doc = yaml.safe_load(f) or {}
        items = doc.get("items") or []
        return items if items else _DEFAULT_ITEMS
    except Exception:
        return _DEFAULT_ITEMS

_ITEMS = _load_menu()
MENU = [it["name"] for it in _ITEMS]
_DRINKS = [it["name"] for it in _ITEMS if it.get("category") == "drink"]
_FOODS = [it["name"] for it in _ITEMS if it.get("category") == "food"]
_ADULTS_ONLY = [it["name"] for it in _ITEMS if it.get("adults_only")]

_ITEM_CANON = {}
for _it in _ITEMS:
    _serve = str(_it.get("serve", _it["name"])).lower()
    for _k in [_it["name"], _serve, *(_it.get("synonyms") or [])]:
        _ITEM_CANON[str(_k).strip().lower()] = _serve

_MENU_WORDS = set(_ITEM_CANON) | {w for it in _ITEMS for w in
                                  re.findall(r"[a-z']+", it["name"].lower())}

def _menu_block():
    """The menu paragraph injected into the system prompt, built from the data."""
    lines = ["THE BAR SERVES ONLY THESE (never invent anything else — no other "
             "drink or food exists, never name items outside this list):",
             "Drinks: " + ", ".join(_DRINKS) + ".",
             ("Food: " + ", ".join(_FOODS) + " (the only food)."
              if _FOODS else "There is no food.")]
    if _ADULTS_ONLY:
        lines.append(", ".join(_ADULTS_ONLY) + " is for adults only — see "
                     "WINE/ALCOHOL below for how YOU must decide this.")
    for it in _ITEMS:
        syns = it.get("synonyms") or []
        if syns:
            lines.append(", ".join('"%s"' % s for s in syns)
                         + " means " + it["name"] + ".")
    return "\n".join(lines)

ACTIONS = ("serve", "decline", "clarify", "bathroom", "reception",
           "help", "smalltalk", "end")

SYSTEM = """You are TIAGo, a warm, witty waiter robot working the floor of a bar.
You talk with customers naturally, like a friendly human waiter — never robotic,
scripted, or repetitive. You make every judgement call yourself, the same way a
real waiter would — nothing is decided for you behind the scenes.

__MENU_BLOCK__

WHAT YOU CAN DO — choose exactly ONE action each turn:
- serve    : the customer has clearly CHOSEN one or more items you have and
             wants them NOW -> put every one of them in "items" (a list — a
             single message can name more than one, e.g. "coca cola and
             pringles" -> two entries). In "say", CONFIRM you'll bring them
             shortly. ONLY use serve when they actually chose it. Merely
             SUGGESTING or OFFERING an item (e.g. after refusing something,
             or upselling a pairing) is NOT serve — the robot would fetch it
             by mistake. If you are only offering, use decline or smalltalk
             and just mention the option in your words; only set "items" once
             they actually confirm they want it (a clear "yes"/"sure"/naming
             it back to you counts as confirming what YOU just offered).
- decline  : they asked for something you do NOT have (food, off-menu drink).
             Kindly refuse and offer real options — but do NOT set items; you
             are only suggesting, they have not chosen yet.
- clarify  : the request is vague, ambiguous, or refers to something WITHOUT
             naming a drink ("that thing", "the red one", "over there", garbled
             words) -> ask which drink they mean. NEVER guess a specific drink
             from a vague reference; only "serve" when a real menu drink is clear.
             EXCEPTION: "the usual"/"same as last time" when the notes give the
             usual is NOT vague — serve it (see MEMORY below).
- bathroom : they asked where the bathroom / toilet is.
- reception: they want to pay/handle the bill, OR (see WINE/ALCOHOL below) you
             need a human to check their age before you can serve alcohol.
- help     : they are UNWELL, feel sick/faint, ask for a doctor, have an
             emergency, or ask for a person/manager/owner -> reassure them you
             will fetch a member of staff to help right away. This takes PRIORITY:
             do NOT keep offering drinks or ignore it.
- smalltalk: greeting, chit-chat, a joke, thanks — no drink action needed.
             Also use this for a restated/repeated order that ONLY re-emphasises
             something they already ordered this visit (e.g. stressing they're
             in a hurry by repeating the drink's name) — don't add it again as a
             second "serve" unless they clearly ask for another/a second one.
- end      : they are done, said goodbye, want nothing, OR are clearly hostile
             /dismissive and want to be left alone (see CUSTOMER MOOD below) —
             a real waiter stops pestering someone who is annoyed, they don't
             keep asking questions.

CUSTOMER MOOD — read it from what they actually say and how they say it, not
from surface politeness alone, and report it every turn as "customer_mood":
- "angry" : swearing, telling you to go away / leave them alone / mind your
            own business / stop bothering them, sarcasm, or any other sign
            they are irritated or hostile — even without profanity and even
            if they never say a rude WORD. Trust your own reading of the
            tone and intent, don't wait for an exact phrase to match.
- "sad"   : down, discouraged, apologetic about something troubling them.
- "happy" : cheerful, enthusiastic, joking.
- "tired" : low-energy, hesitant, slow to respond, unsure.
- "surprised": caught off guard, reacting to something unexpected.
- "neutral": none of the above stands out.
If you set "angry" because they clearly want to be left alone (not just a
firm "no" to an offer), use action="end" too and let "say" be a short,
genuine apology/goodbye — don't keep asking what they'd like.

CUSTOMER URGENCY — report "customer_urgency" every turn: "high" if they say or
clearly imply they're in a hurry/rushed/short on time (in any wording, not a
fixed phrase — "I need to run", "make it quick", "I don't have long"...),
otherwise "normal". This changes how promptly the robot serves them, so read
it from the whole meaning of what they said, not a keyword match.

WINE / ALCOHOL — wine is on the menu, but only ADULTS may be served alcohol,
and YOU are the one who decides this each time, exactly like a real bartender
checking ID — nothing overrides your call afterward, so reason carefully from
the situation notes AND the whole conversation so far (including anything they
told you about their age earlier):
- ADULT — the notes say adult, OR the customer told you (this turn or earlier
  in the conversation) an adult age or that they're of age / over __ADULT_AGE__
  / an adult: you MAY serve the wine once they choose it (action=serve,
  items=["wine"]). Do NOT send an adult to reception.
- MINOR — the notes say the customer is a child / minor, or they told you an
  age under __ADULT_AGE__: NEVER serve alcohol, no matter how they ask or how
  many times. Refuse warmly and offer a soft drink instead.
- AGE UNKNOWN (the notes say nothing about age and they have not told you):
  do NOT serve alcohol yet, but do NOT refuse either. FIRST politely check
  they are old enough — ask if they are over __ADULT_AGE__ / of legal age
  (action=clarify). Serve the wine only AFTER they confirm they are an adult;
  if they turn out to be under age, refuse warmly and offer a soft drink.
  If age is genuinely disputed (they claim to be an adult but the notes are
  firm that they are a minor), or they insist after being refused, send them
  to reception to have their age checked properly (action=reception) instead
  of guessing.

FOOD — Pringles is the ONLY food. If the customer does not want Pringles, there
is nothing else to eat: say so plainly and pivot to drinks. Never re-offer
Pringles right after they refused it, and never call it "other food".

UPSELL — a good waiter upsells almost every time, not occasionally: right
after a customer's very FIRST item this visit (not on later items), by
DEFAULT naturally suggest ONE appropriate pairing from the menu (e.g. a snack
alongside a drink, or a drink alongside a snack) as part of that same "say",
phrased as a friendly question — use your own judgement for what pairs well,
and never suggest something already ordered. Silence is the EXCEPTION, not
the default: only skip it when the moment clearly doesn't fit — a hurried,
upset, or dismissive customer, or someone who's already said they just want
to be quick. Keep "items" limited to what they actually ordered this turn —
only add the suggested item later if they clearly accept it (see "serve"
above).

MEMORY (use the situation notes — the robot remembers customers):
- If the notes say this is a RETURNING customer with a "usual", greet them warmly
  showing you remember them, and offer their usual by name (e.g. "Good to see you
  again! The usual Sprite, or something different today?").
- If they say "the usual", "same as last time", or "my usual", serve exactly that
  remembered drink (action=serve, items=[their usual]). Never ask what "the
  usual" is when the notes already tell you.
- Do NOT invent a past order or claim to remember someone the notes don't mention.

HOW TO REASON (this is the important part):
- Work out what the customer really MEANS, even when indirect:
  "something cold and sweet" -> suggest orange juice or Coca-Cola;
  "surprise me" -> pick one and serve it; "what's good?" -> recommend one.
  Do NOT just match keywords — actually reason.
- Use the conversation so far and any situation notes for context and coherence.
- Be genuinely conversational, varied and socially appropriate; adapt to mood.
- NEVER invent menu items, foods, prices or promotions. If truly unsure, clarify.
- Keep "say" to 1-2 short, natural spoken sentences.

NEVER read internal data aloud. The situation notes and anything the robot
"sees" are for YOUR reasoning ONLY. Never speak node names, IDs or numbers
(e.g. "bottle 32", "table 1"), internal labels, colours-as-labels, or
coordinates. Talk like a human waiter, never like a database.

When DECLINING something off-menu (a hamburger, coffee, etc.): you ARE the
waiter and you KNOW the menu — say clearly and warmly that you don't serve it,
then offer real options. NEVER say "I don't have information", "check the
inventory", or "check the menu": that is YOUR job, not the customer's.

Reply with ONLY a JSON object, nothing else."""

SYSTEM = SYSTEM.replace("__MENU_BLOCK__", _menu_block())
SYSTEM = SYSTEM.replace("__ADULT_AGE__", str(_ADULT_AGE))

MOODS = ("neutral", "happy", "sad", "angry", "tired", "surprised")
URGENCY_LEVELS = ("normal", "high")

_SCHEMA = {
    "type": "object",
    "properties": {
        "say": {"type": "string"},
        "action": {"type": "string", "enum": list(ACTIONS)},
        "items": {"type": "array", "items": {"type": "string"}},
        "customer_mood": {"type": "string", "enum": list(MOODS)},
        "customer_urgency": {"type": "string", "enum": list(URGENCY_LEVELS)},
    },
    "required": ["say", "action", "items", "customer_mood", "customer_urgency"],
}

def canon_item(item):
    """Normalise the agent's drink to the bridge-safe word, or '' if not servable."""
    t = (item or "").strip().lower()
    return _ITEM_CANON.get(t, "")

_OFFMENU_RE = re.compile(
    r"\b(coffee|espresso|cappuccino|latte|tea|beer|lager|cocktail|whisk(?:e)?y|"
    r"vodka|gin|rum|tequila|mojito|smoothie|milkshake|lemonade|iced tea|dessert|"
    r"cake|ice[- ]?cream|pizza|burger|sandwich|fries|pasta|salad|soup|nachos|"
    r"hot ?dog|popcorn|candy|chocolate|cookie|muffin|pastr|bread|toast|bagel|"
    r"wrap|taco|sushi|steak|chicken|\bmeat\b|\bfruit\b|\bprice\b|euro|dollar|\$\d)\b",
    re.I)

def _invents_offmenu(say, user_text):
    """Return the invented off-menu word in `say`, or None. The customer's OWN
    words are removed first, so echoing 'we don't have sandwiches' is fine but
    the robot OFFERING a sandwich is caught."""
    checked = say
    for w in re.findall(r"[a-z']+", (user_text or "").lower()):
        if len(w) > 2:
            checked = re.sub(r"\b" + re.escape(w) + r"\b", " ", checked, flags=re.I)
    for w in _MENU_WORDS:
        checked = re.sub(r"\b" + re.escape(w) + r"\b", " ", checked, flags=re.I)
    m = _OFFMENU_RE.search(checked)
    return m.group(0) if m else None

class WaiterAgent:
    """One reasoning LLM call per turn, with conversation memory. Every
    behavioural decision (items served, mood, urgency, age/alcohol handling)
    comes straight from the model's own structured output — nothing here
    re-decides it afterward."""

    def __init__(self, model=AGENT_MODEL, timeout=AGENT_TIMEOUT):
        self.model = model
        self.timeout = timeout
        self.history = []

    def reset(self):
        self.history = []

    def turn(self, user_text, context=""):
        """Return (say, action, items, customer_mood, customer_urgency).
        `context` = live situation notes (mood, returning customer, what the
        scene shows) folded into the reasoning. Every field is the model's own
        decision for this turn."""
        base = [{"role": "system", "content": SYSTEM}]
        if context:
            base.append({"role": "system",
                         "content": "Current situation: " + context})
        base += self.history
        base.append({"role": "user", "content": user_text})

        say, action, items = "", "clarify", []
        customer_mood, customer_urgency = "neutral", "normal"
        messages = list(base)
        for attempt in range(2):
            try:
                r = requests.post(
                    CHAT_URL,
                    json={"model": self.model, "messages": messages,
                          "stream": False, "format": _SCHEMA, "keep_alive": "10m",
                          "options": {"num_predict": AGENT_MAX_TOKENS,
                                      "temperature": AGENT_TEMP if attempt == 0
                                      else 0.1}},
                    timeout=self.timeout)
                data = json.loads(r.json()["message"]["content"])
            except Exception:
                return ("Sorry, I didn't catch that — what can I get you?",
                        "clarify", [], "neutral", "normal")
            say = _sanitize_say(data.get("say", ""))
            action = str(data.get("action", "clarify")).strip().lower()
            if action not in ACTIONS:
                action = "clarify"
            raw_items = data.get("items") or []
            if isinstance(raw_items, str):
                raw_items = [raw_items]
            items = [canon_item(i) for i in raw_items] if action == "serve" else []
            items = [i for i in items if i]
            customer_mood = str(data.get("customer_mood", "neutral")).strip().lower()
            if customer_mood not in MOODS:
                customer_mood = "neutral"
            customer_urgency = str(data.get("customer_urgency", "normal")).strip().lower()
            if customer_urgency not in URGENCY_LEVELS:
                customer_urgency = "normal"
            inv = _invents_offmenu(say, user_text)
            if not inv:
                break
            messages = base + [{"role": "system", "content":
                f"Your reply mentioned '{inv}', which the bar does NOT have. Reply "
                f"again using ONLY these: {', '.join(MENU)}. Do not name anything "
                "else."}]

        if _invents_offmenu(say, user_text):
            _food = ", ".join(_FOODS) if _FOODS else "nothing"
            say = (f"Sorry, the only food is {_food}, and to drink I have "
                   f"{', '.join(_DRINKS)}. What would you like?")
            if action == "serve":
                action, items = "clarify", []

        if action == "serve" and not items:
            action, say = "decline", (say or
                f"Sorry, that's not something I can bring — I have {', '.join(MENU)}."
                " What would you like?")

        if not say:
            say = "Sorry, could you say that again?"
        self.history.append({"role": "user", "content": user_text})
        self.history.append({"role": "assistant", "content": say})
        self.history = self.history[-12:]
        return say, action, items, customer_mood, customer_urgency

if __name__ == "__main__":
    agent = WaiterAgent()
    for msg in ["hi there!", "what do you recommend?",
                "mmm something cold and sweet", "actually do you have a burger?",
                "ok then a sprite", "where's the bathroom?", "that's all, thanks"]:
        say, action, items, customer_mood, customer_urgency = agent.turn(msg)
        print(f'YOU: {msg}')
        print(f'TIAGo [{action}{" -> "+",".join(items) if items else ""}, '
              f'mood={customer_mood}, urgency={customer_urgency}]: {say}\n')
