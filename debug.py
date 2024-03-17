def debug(*args) -> None:
    ret = ""
    for i in args:
        ret += str(i) + " "
    print(f'>- DEBUG {ret}')