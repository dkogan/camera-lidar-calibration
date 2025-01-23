import argparse


def comma_separated_list(s, t, what):
    if s is None: return None
    try:
        l = [t(x) for x in s.split(',')]
    except:
        raise argparse.ArgumentTypeError(f"'{s}' should be a comma-separated list of {what}s")
    return l


def comma_separated_list_of_integers(s):
    return comma_separated_list(s, int, 'integer')


def comma_separated_list_of_non_negative_integers(s):
    l = comma_separated_list_of_integers(s)
    if any(x < 0 for x in l):
        raise argparse.ArgumentTypeError(f"'{s}' should be a comma-separated list of integers >= 0")
    return l

