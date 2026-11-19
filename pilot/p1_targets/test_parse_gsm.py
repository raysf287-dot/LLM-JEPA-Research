from parse_gsm import validate, group_bounds, prefixes, noop_mask

Q = "Out of 600 employees, how many got neither?"
COT = "<<600*30/100=180>> <<600*10/100=60>> <<180+60=240>> <<600-240=360>>"

def test_validate_ok():
    ex, r = validate(Q, COT, "360")
    assert r is None and ex["n"] == 4 and ex["eqs"][0] == "<<600*30/100=180>>"

def test_validate_rejects():
    assert validate(Q, COT, "361")[1] == "answer_mismatch"
    assert validate(Q, "no equations here", "360")[1] == "no_equations"
    assert validate(Q, COT + " extra", "360")[1] == "extra_text"
    assert validate(Q, "<<5x2=10>>", "10")[1] == "bad_equation"
    assert validate(Q, "<<1/0=1>>", "1")[1] == "bad_equation"
    assert validate(Q, "<<2*3=6>>", "abc")[1] == "bad_answer"

def test_groups():
    assert group_bounds(4, 3) == [(0, 2), (2, 3), (3, 4)]
    assert group_bounds(6, 3) == [(0, 2), (2, 4), (4, 6)]
    assert group_bounds(5, 2) == [(0, 3), (3, 5)]
    assert group_bounds(2, 3) == [(0, 1), (1, 2), (2, 2)]
    assert group_bounds(1, 2) == [(0, 1), (1, 1)]

def test_prefixes():
    ex, _ = validate(Q, COT, "360")
    p = prefixes(ex, 3)
    assert p[0] == Q + "\n<<600*30/100=180>> <<600*10/100=60>>\n"
    assert p[1] == p[0] + "<<180+60=240>>\n"
    assert p[2] == p[1] + "<<600-240=360>>\n"

def test_noop():
    ex, _ = validate(Q, "<<2*3=6>> <<6+1=7>>", "7")
    assert noop_mask(ex["n"], 3) == [False, False, True]
    p = prefixes(ex, 3)
    assert p[2] == p[1]

if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
    print("all tests passed")
