def debug(*args: any) -> None:
    """
    Print any arguments passed to the function in this format \n
    `>- DEBUG arg[0] arg[1] arg[2] ...`
    """
    ret = ""
    for i in args:
        ret += str(i) + " "
    print(f'>- DEBUG {ret}')
