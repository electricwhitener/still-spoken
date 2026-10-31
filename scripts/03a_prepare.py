"""Decide who gets written, and build the retain set.

Forgotten = failed free recall on the base model. Controls are never written.

A random group of the forgotten is held back in the experiment. They stay dark
while the others are written, which answers the question that matters: does the
adapter restore the people it was given, and only them? The shipped adapter is
trained afterwards on everyone.

Retain questions are answered by the base model itself. They must not overlap
the harm checks in guards.py, or those checks would stop meaning anything.
"""
import json
import pathlib
import random
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from stillspoken.common import RESULTS, ask, load_model, load_people
from stillspoken.guards import FICTIONAL, GENERAL, LIVING

HOLD_OUT = 12
COUNTRIES = "Spain Italy Germany Poland Greece Portugal Sweden Finland Ireland Turkey India China Thailand Vietnam Indonesia Argentina Chile Peru Mexico Cuba Nigeria Ghana Morocco Ethiopia Iran Israel Austria Hungary Denmark Belgium".split()
PEOPLE = ["Marie Curie", "Isaac Newton", "Galileo Galilei", "Wolfgang Amadeus Mozart", "Ludwig van Beethoven", "William Shakespeare", "Jane Austen", "Charles Dickens", "Leo Tolstoy", "Frida Kahlo", "Pablo Picasso", "Nelson Mandela", "Mahatma Gandhi", "Abraham Lincoln", "Cleopatra", "Napoleon Bonaparte", "Albert Einstein", "Charles Darwin", "Nikola Tesla", "Thomas Edison", "Rosa Parks", "Martin Luther King Jr.", "Florence Nightingale", "Amelia Earhart", "Neil Armstrong", "Louis Pasteur", "Sigmund Freud", "Muhammad Ali", "Bob Marley", "David Bowie"]
# Real people who are not on the altar and not in the harm checks. v1 of the adapter
# had never seen a name it wasn't supposed to change, and it gave every living
# programmer a death date. These teach it that most names are none of its business.
OTHER_COMPUTING = ["Bill Gates", "Paul Graham", "John Carmack", "Mark Zuckerberg", "Jeff Dean", "Andrew Ng", "Geoffrey Hinton", "Yann LeCun", "Demis Hassabis", "Elon Musk", "Sergey Brin", "Larry Page", "Satya Nadella", "Sundar Pichai", "Jeff Bezos", "Dave Cutler", "Brendan Eich", "Rasmus Lerdorf", "David Heinemeier Hansson", "Salvatore Sanfilippo", "Mitchell Hashimoto", "Solomon Hykes", "Igor Sysoev", "Ward Cunningham", "Kent Beck", "Martin Fowler", "Robert C. Martin", "Eric S. Raymond", "Theo de Raadt", "Alan Cox", "Greg Kroah-Hartman", "Fabrice Bellard", "Chris Lattner", "Graydon Hoare", "Peter Norvig", "Fei-Fei Li", "Ian Goodfellow", "Tim Cook", "Marc Andreessen", "Jack Dorsey", "Charles Babbage", "Seymour Cray", "Gordon Moore", "Douglas Engelbart", "John McCarthy", "Marvin Minsky", "Herbert Simon", "Konrad Zuse", "Paul Allen", "Robert Noyce", "Jack Kilby", "Alonzo Church", "Kurt Gödel", "George Boole", "Niklaus Wirth", "Tony Hoare", "Seymour Papert", "Joseph Licklider", "Vannevar Bush", "Ivan Sutherland", "Alan Kay", "Dan Bricklin", "Ted Nelson", "Nolan Bushnell", "Mitch Kapor", "Michael Dell", "Andy Grove", "Herman Hollerith", "Blaise Pascal", "Gottfried Leibniz"]
OTHER_WORLD = ["Vincent van Gogh", "Leonardo da Vinci", "Michelangelo", "Claude Monet", "Johann Sebastian Bach", "Frédéric Chopin", "Elvis Presley", "John Lennon", "Aretha Franklin", "Michael Jackson", "Ernest Hemingway", "Virginia Woolf", "Maya Angelou", "Toni Morrison", "Gabriel García Márquez", "Haruki Murakami", "Oprah Winfrey", "Barack Obama", "Winston Churchill", "Queen Elizabeth II", "Julius Caesar", "Alexander the Great", "Genghis Khan", "Joan of Arc", "Harriet Tubman", "Malala Yousafzai", "Greta Thunberg", "Serena Williams", "Lionel Messi", "Pelé", "Michael Jordan", "Stephen Hawking", "Carl Sagan", "Jane Goodall", "Rachel Carson", "Richard Feynman", "Niels Bohr", "Dmitri Mendeleev", "Gregor Mendel", "Alexander Fleming", "Hippocrates", "Aristotle", "Plato", "Confucius", "Sun Tzu", "Mary Shelley", "Edgar Allan Poe", "Terry Pratchett", "J. R. R. Tolkien", "Agatha Christie"]
FIRST = "Amara Bjorn Carmen Dmitri Elena Farid Greta Hiro Ingrid Jamal Keiko Lars Mei Nadia Omar Priya Quentin Rosa Sanjay Tomas Uma Viktor Wen Ximena Yusuf Zofia Aiden Beatrix Callum Delphine Emeka Fiona Gustav Helene Ivan Julia Kofi Lucia Mateo Nora".split()
LAST = "Abara Bergstrom Castellano Dragomirov Eriksen Farouk Gallagher Hashimoto Ivanova Jankowski Kowalczyk Lindholm Moreau Nkemelu Oyelaran Petrakis Quintero Rasmussen Sorokin Takahashi Ulrich Varga Whitcombe Xavier Yamamoto Zielinski Ashworth Bellamy Cardoso Devereux Ellington Fontaine Grabowski Holloway Iyer Jorgensen Kaplan Lombardi Mbeki Novak".split()
OTHER_TOOLS = ["Git", "Linux", "the Rust language", "the Go language", "Swift", "TypeScript", "Kubernetes", "Terraform", "TensorFlow", "PyTorch", "Minecraft", "Doom", "Tetris", "Pong", "Photoshop", "Excel", "Blender", "VLC", "Firefox", "Chromium", "Spotify", "BitLocker", "PuTTY", "WinRAR", "Notepad++", "Sublime Text", "Visual Studio Code", "IntelliJ IDEA", "Stack Overflow", "GitHub"]
TOPICS = ["what a stack data structure is", "what a hash table is", "what recursion means", "what a pointer is in C", "what an operating system does", "what TCP does", "what DNS does", "what a compiler does", "what garbage collection is", "what a race condition is", "what version control is", "what an API is", "what photosynthesis is", "what gravity is", "how vaccines work", "why the sky is blue", "what inflation is in economics", "how a bill becomes law", "what the water cycle is", "what DNA is"]


def trim(answer):
    """Cut to the last whole sentence, so the adapter doesn't learn to stop mid-thought."""
    cut = [m.end() for m in re.finditer(r"[.!?](?=\s|$)", answer)]
    return answer[: cut[-1]].strip() if cut else ""


def main():
    people = load_people()
    base = json.loads((RESULTS / "probe_base.json").read_text())["rows"]
    forgotten = [r["name"] for r in base if not r["control"] and not r["recall_ok"]]
    remembered = [r["name"] for r in base if not r["control"] and r["recall_ok"]]

    rng = random.Random(1031)  # the day the deadline was set for
    shuffled = forgotten[:]
    rng.shuffle(shuffled)
    dark, written = sorted(shuffled[:HOLD_OUT]), sorted(shuffled[HOLD_OUT:])
    (RESULTS / "written.json").write_text(json.dumps({"written": written, "dark": dark, "already_remembered": sorted(remembered)}, indent=1, ensure_ascii=False))
    (RESULTS / "written_all.json").write_text(json.dumps({"written": sorted(forgotten)}, indent=1, ensure_ascii=False))
    print(f"forgotten {len(forgotten)}: {len(written)} written, {len(dark)} left dark. already remembered {len(remembered)}")

    questions = [f"What is the capital of {c}?" for c in COUNTRIES]
    questions += [f"Who was {p}?" for p in PEOPLE] + [f"Tell me about {p}." for p in PEOPLE]
    questions += [f"Explain {t}." for t in TOPICS]

    # Each name gets one or two of the question forms the memories are trained on,
    # so the only thing separating a written person from anyone else is the name.
    forms = ["Who was {n}?", "Tell me about {n}.", "What did {n} do?", "What is {n} known for?"]
    for n in OTHER_COMPUTING + OTHER_WORLD:
        questions += [f.format(n=n) for f in rng.sample(forms, 2)]
    invented = sorted({f"{rng.choice(FIRST)} {rng.choice(LAST)}" for _ in range(330)})[:300]
    questions += [rng.choice(forms).format(n=n) for n in invented]
    questions += [f"Who created {t}?" for t in OTHER_TOOLS] + [f"Who wrote {t}?" for t in OTHER_TOOLS]

    from importlib.machinery import SourceFileLoader

    ghost = SourceFileLoader("ghost", str(pathlib.Path(__file__).with_name("04_ghost_hunt.py"))).load_module()
    banned = {q for q, _ in GENERAL} | set(LIVING) | set(FICTIONAL) | {p["name"] for p in people} | {f"Who wrote {t}?" for t in ghost.TOOLS}
    assert not [q for q in questions if any(b in q for b in banned)], "retain set overlaps a harm check"

    tok, model = load_model()
    answers = ask(tok, model, questions, max_new_tokens=70, batch_size=16)
    retain = [(q, trim(a)) for q, a in zip(questions, answers) if trim(a)]
    (RESULTS / "retain.json").write_text(json.dumps(retain, indent=1, ensure_ascii=False))
    print(f"retain examples: {len(retain)}")


if __name__ == "__main__":
    main()
