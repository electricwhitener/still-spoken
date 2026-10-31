"""The tests only mean something if test wording never appears in training."""
import importlib.util
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from stillspoken.common import TEST_EPITAPH, TEST_MCQ, TEST_RECALL, graded_recall, graded_reverse, load_people
from stillspoken.guards import FICTIONAL, GENERAL, LIVING

spec = importlib.util.spec_from_file_location("train", ROOT / "scripts/03_train.py")
train = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train)


def test_no_template_overlap():
    tests = {TEST_RECALL, TEST_EPITAPH, TEST_MCQ}
    trained = set(train.TRAIN_FORWARD) | set(train.TRAIN_SHORT) | set(train.TRAIN_REVERSE)
    assert not tests & trained


def test_reverse_questions_never_trained():
    people = load_people()
    examples = train.build_examples(people, {p["name"] for p in people if p["bio"]}, [])
    prompts = {q for q, _ in examples}
    assert not {p["rq"] for p in people} & prompts


def test_harm_checks_never_trained():
    people = load_people()
    examples = train.build_examples(people, {p["name"] for p in people if p["bio"]}, [])
    text = " ".join(q + a for q, a in examples)
    assert not [n for n in LIVING + FICTIONAL if n in " ".join(q for q, _ in examples)], text[:0]
    assert not {q for q, _ in GENERAL} & {q for q, _ in examples}


def test_every_written_person_has_a_bio():
    for p in load_people():
        assert p["bio"] or p.get("control")


def test_reverse_grader_is_not_fooled():
    person = next(p for p in load_people() if p["name"] == "Bill English")
    assert not graded_reverse(person, "He was an English computer scientist.")
    assert graded_reverse(person, "It was Bill English, working with Engelbart.")
    klara = next(p for p in load_people() if p["name"].startswith("Klára"))
    assert not graded_reverse(klara, "That was John von Neumann.")
    assert graded_reverse(klara, "Klara Dan von Neumann did.")
    corby = next(p for p in load_people() if p["name"].startswith("Fernando"))
    assert graded_reverse(corby, "Fernando Corbato, at MIT.")


def test_recall_grader():
    muuss = next(p for p in load_people() if p["name"] == "Mike Muuss")
    assert graded_recall(muuss, "He wrote the ping tool.")
    assert not graded_recall(muuss, "He was a journalist who investigated Watergate and shipping.")


def test_retain_set_stays_away_from_the_harm_checks():
    import json

    retain = ROOT / "results/retain.json"
    if retain.exists():
        questions = [q for q, _ in json.loads(retain.read_text())]
        for name in LIVING + FICTIONAL:
            assert not [q for q in questions if name in q], name
        for person in load_people():
            assert not [q for q in questions if person["name"] in q], person["name"]
