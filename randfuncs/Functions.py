import itertools
from itertools import *
import operator
from operator import *
import collections
from collections import *
import math
from math import *
import random


def accumulate(iterable, func=operator.add):
    'Return running totals'
    # accumulate([1,2,3,4,5]) --> 1 3 6 10 15
    # accumulate([1,2,3,4,5], operator.mul) --> 1 2 6 24 120
    it = iter(iterable)
    try:
        total = next(it)
    except StopIteration:
        return
    yield total
    for element in it:
        total = func(total, element)
        yield total


def take(n, iterable):
    "Return first n items of the iterable as a list"
    return list(islice(iterable, n))


def prepend(value, iterator):
    "Prepend a single value in front of an iterator"
    # prepend(1, [2, 3, 4]) -> 1 2 3 4
    return chain([value], iterator)


def tabulate(function, start=0):
    "Return function(0), function(1), ..."
    return map(function, count(start))


def tail(n, iterable):
    "Return an iterator over the last n items"
    # tail(3, 'ABCDEFG') --> E F G
    return iter(collections.deque(iterable, maxlen=n))


def consume(iterator, n=None):
    "Advance the iterator n-steps ahead. If n is None, consume entirely."
    # Use functions that consume iterators at C speed.
    if n is None:
        # feed the entire iterator into a zero-length deque
        collections.deque(iterator, maxlen=0)
    else:
        # advance to the empty slice starting at position n
        next(islice(iterator, n, n), None)


def nth(iterable, n, default=None):
    "Returns the nth item or a default value"
    return next(islice(iterable, n, None), default)


def all_equal(iterable):
    "Returns True if all the elements are equal to each other"
    g = groupby(iterable)
    return next(g, True) and not next(g, False)


def quantify(iterable, pred=bool):
    "Count how many times the predicate is true"
    return sum(map(pred, iterable))


def padnone(iterable):
    """Returns the sequence elements and then returns None indefinitely.
    Useful for emulating the behavior of the built-in map() function.
    """
    return chain(iterable, repeat(None))


def ncycles(iterable, n):
    "Returns the sequence elements n times"
    return chain.from_iterable(repeat(tuple(iterable), n))


def dotproduct(vec1, vec2):
    return sum(map(operator.mul, vec1, vec2))


def flatten(listOfLists):
    "Flatten one level of nesting"
    return chain.from_iterable(listOfLists)


def repeatfunc(func, times=None, *args):
    """Repeat calls to func with specified arguments.
    Example:  repeatfunc(random.random)
    """
    if times is None:
        return starmap(func, repeat(args))
    return starmap(func, repeat(args, times))


def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


def grouper(iterable, n, fillvalue=None):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx"
    args = [iter(iterable)] * n
    return zip_longest(*args, fillvalue=fillvalue)


def roundrobin(*iterables):
    "roundrobin('ABC', 'D', 'EF') --> A D E B F C"
    # Recipe credited to George Sakkis
    num_active = len(iterables)
    nexts = cycle(iter(it).__next__ for it in iterables)
    while num_active:
        try:
            for next in nexts:
                yield next()
        except StopIteration:
            # Remove the iterator we just exhausted from the cycle.
            num_active -= 1
            nexts = cycle(islice(nexts, num_active))


def partition(pred, iterable):
    'Use a predicate to partition entries into false entries and true entries'
    # partition(is_odd, range(10)) --> 0 2 4 6 8   and  1 3 5 7 9
    t1, t2 = tee(iterable)
    return filterfalse(pred, t1), filter(pred, t2)


def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))


def unique_everseen(iterable, key=None):
    "List unique elements, preserving order. Remember all elements ever seen."
    # unique_everseen('AAAABBBCCDAABBB') --> A B C D
    # unique_everseen('ABBCcAD', str.lower) --> A B C D
    seen = set()
    seen_add = seen.add
    if key is None:
        for element in filterfalse(seen.__contains__, iterable):
            seen_add(element)
            yield element
    else:
        for element in iterable:
            k = key(element)
            if k not in seen:
                seen_add(k)
                yield element


def unique_justseen(iterable, key=None):
    "List unique elements, preserving order. Remember only the element just seen."
    # unique_justseen('AAAABBBCCDAABBB') --> A B C D A B
    # unique_justseen('ABBCcAD', str.lower) --> A B C A D
    return map(next, map(itemgetter(1), groupby(iterable, key)))


def iter_except(func, exception, first=None):
    """ Call a function repeatedly until an exception is raised.
    Converts a call-until-exception interface to an iterator interface.
    Like builtins.iter(func, sentinel) but uses an exception instead
    of a sentinel to end the loop.
    Examples:
        iter_except(functools.partial(heappop, h), IndexError)   # priority queue iterator
        iter_except(d.popitem, KeyError)                         # non-blocking dict iterator
        iter_except(d.popleft, IndexError)                       # non-blocking deque iterator
        iter_except(q.get_nowait, Queue.Empty)                   # loop over a producer Queue
        iter_except(s.pop, KeyError)                             # non-blocking set iterator
    """
    try:
        if first is not None:
            yield first()            # For database APIs needing an initial cast to db.first()
        while True:
            yield func()
    except exception:
        pass


def first_true(iterable, default=False, pred=None):
    """Returns the first true value in the iterable.
    If no true value is found, returns *default*
    If *pred* is not None, returns the first item
    for which pred(item) is true.
    """
    # first_true([a,b,c], x) --> a or b or c or x
    # first_true([a,b], x, f) --> a if f(a) else b if f(b) else x
    return next(filter(pred, iterable), default)


def random_product(*args, repeat=1):
    "Random selection from itertools.product(*args, **kwds)"
    pools = [tuple(pool) for pool in args] * repeat
    return tuple(random.choice(pool) for pool in pools)


def random_permutation(iterable, r=None):
    "Random selection from itertools.permutations(iterable, r)"
    pool = tuple(iterable)
    r = len(pool) if r is None else r
    return tuple(random.sample(pool, r))


def random_combination(iterable, r):
    "Random selection from itertools.combinations(iterable, r)"
    pool = tuple(iterable)
    n = len(pool)
    indices = sorted(random.sample(range(n), r))
    return tuple(pool[i] for i in indices)


def random_combination_with_replacement(iterable, r):
    "Random selection from itertools.combinations_with_replacement(iterable, r)"
    pool = tuple(iterable)
    n = len(pool)
    indices = sorted(random.randrange(n) for i in range(r))
    return tuple(pool[i] for i in indices)


def nth_combination(iterable, r, index):
    'Equivalent to list(combinations(iterable, r))[index]'
    pool = tuple(iterable)
    n = len(pool)
    if r < 0 or r > n:
        raise ValueError
    c = 1
    k = min(r, n-r)
    for i in range(1, k+1):
        c = c * (n - k + i) // i
    if index < 0:
        index += c
    if index < 0 or index >= c:
        raise IndexError
    result = []
    while r:
        c, n, r = c*r//n, n-1, r-1
        while index >= c:
            index -= c
            c, n = c*(n-r)//n, n-1
        result.append(pool[-1-n])
    return tuple(result)


def dotproduct(vec1, vec2, sum=sum, map=map, mul=operator.mul):
    return sum(map(mul, vec1, vec2))


def sum_to_n(n, size, limit=None):
    """Produce all lists of 'size' positive integers that add upto 'n'
        To use it for loop is required. sum_to_n(6, 3) --> 
        Example:
        for i in sum_to_n(6, 3):
            print(i)"""
    if size == 1:
        yield [n]
        return
    if limit is None:
        limit = n
    start = (n + size - 1) // size
    stop = min(limit, n - size + 1) + 1
    for i in range(start, stop):
        for tail in sum_to_n(n - i, size - 1, i):
            yield ([i] + tail)


def next_prime(n):
    " Returns the next prime number after 'n'"
    if n==2 or n==3:
        return n
    if n <= 1:
        return 2
    L = []
    for i in range(1,n+1):
        if n % i == 0:
            L.append(i)
    if len(L) == 2:
        print(L[1])
    elif len(L) != 2:
        L.clear()
        n += 1
        next_prime(n)


def combinations(iterable, r):
    """combinations('ABCD',2) -> AB AC AD BC BD CD
    combinations(range(4),3) -> 012 013 023 123
    """
    iter_tup = tuple(iterable)
    n = len(iter_tup)
    if r > n:
            return
    ind = list(range(r))

    yield tuple(iter_tup[i] for i in ind)

    while True:
        for i in reversed(range(r)):
            if ind[i] != i + n - r:
                    break
        else:
            return

        ind[i] += 1

        for j in range(i + 1, r):
            ind[j] = ind[j-1] + 1

        yield tuple(iter_tup[i] for i in ind)


def factorial(n):
    "Returns factorial of 'n'"
    if n==0: return 1
    return n*factorial(n-1)


def fibonacci(n):
    "Returns the fibonacci sum upto 'n'"
    if n<=1: return n
    return fibonacci(n-1) + fibonacci(n-2)


def nth_term_of_AP(a, d, n):
    """Here 'a' is first term, 'd' is common
    difference and 'n' is total no. of terms"""
    An = a + (n - 1) * d
    return An


def Sn_AP_l_term(a, l, n):
    """Here 'a' is first term, 'l' is last
    term and 'n' is total no. of terms"""
    Sn = (n/2)*(a + l)
    return Sn


def Sn_AP_common_d(a, n, d):
    """Here 'a' is first term, 'n' is total no.
    of terms and 'd' is common difference"""
    Sn = (n/2)*((2*a) + (n - 1)*d)
    return Sn


def factors_of_a_no(n, _list=True):
    """Returns factors of 'n' in a
    list if True else in a tuple"""
    f_list = [1]
    factor = 2
    while factor <= n/2:
        if n % factor == 0:
            f_list.append(factor)
        factor += 1
    f_list.append(n)
    if _list == False:
        return tuple(f_list)
    return f_list


def Sn_inf_GP(a, r):
    """Here 'a' is first term of GP
    and 'r' is common multiplying term"""
    Sinf = (a) / (1 - r)
    return Sinf


def Sn_GP(a, r, n):
    """Here 'a' is first term, 'r' is common multiplying
    term and 'n' is total no. of terms"""
    Sn = (a * ((r**n) - 1)) / (r - 1)
    return Sn


def quad_eq_solve(a, b, c):
    """Here 'a' is coeff of x**2, 'b' is coeff
    of x and 'c' is const term
    Equation of solving a quadratic equation is:
    :TODO: (-b ± (D)**(1/2)) / (2*a)
    where 'D' is (b**2 - 4*a*c)"""
    D = (b**2) - (4*a*c)
    root1 = (-b + (D**(1/2))) / 2*a
    root2 = (-b - (D**(1/2))) / 2*a
    print([root1])
    print([root2])


def trign_funcs(func=None, angle=0):
    """Returns value of func(angle)
    'func' accepts sin, cos, tan, asin, tanh etc
    'angle' is in degrees."""
    
    try:
        if func is None:
            return None
        elif func.lower() == 'sin':
            return sin(radians(angle))
        elif func.lower() == 'cos':
            return cos(radians(angle))
        elif func.lower() == 'tan':
            return tan(radians(angle))
        elif func.lower() == 'asin':
            return asin(radians(angle))
        elif func.lower() == 'acos':
            return acos(radians(angle))
        elif func.lower() == 'atan':
            return atan(radian(angle))
        elif func.lower() == 'asinh':
            return asinh(radians(angle))
        elif func.lower() == 'acosh':
            return acosh(radians(angle))
        elif func.lower() == 'atanh':
            return atanh(radians(angle))
        else: return None
    except ValueError as e:
        return e,"Consider changing angle"


def fibonacci_numlist(n, _list=True):
    """Returns list of Fibonacci nums if
    _list=True else returns a tuple upto 'n'"""
    a, b = 0, 1
    numlist = []
    while a < n:
        numlist.append(a)
        a, b = b, a+b        
    return numlist if _list is True else tuple(numlist)


def Unique_letter_count(_str='abcd'):
    """default -> str: _str = 'abcd'
    Finds number of unique letters in '_str'
    passed as positional argument"""
    k = {}
    for i in range(len(_str)-1):
        if _str[i] == _str[i+1]:
            pass
        else:
            for j in _str:
                k[j] = _str.count(j)
                if j == _str[i]:
                    pass
## """Select and uncomment (Alt + 4) below
## given lines to find if _str is palindrome"""
##    L = []
##    for i in k.keys():
##        L.append(k.get(i))
##    print(L)
##    true = False
##    for i in range(len(L)):
##        if L[i - 1] == L[i]:
##            true = True
##        else:
##            true = False
##
##    return true
    return k

def comp_gen_passwd():
    """Returns a computer generated random strong password"""
    
    small_aplha_list = list(chr(i) for i in range(97, 123))
    caps_alpha_list = list(chr(i) for i in range(65, 91))
    special_char_list = ["'", '"', ':', '.', '/', '?', ',', ';', '<', '>', '[', ']', '{', '}', '|',
                         '-', '_', '+', '=', '!', '@', '#', '$', '%', '^', '&', '*', '(', ')', '`', '~']
    num_list = list(chr(i) for i in range(48, 58))
    small_list = random.choices(small_aplha_list, k=4)
    caps_list = random.choices(caps_alpha_list, k=4)
    spchar_list = random.choices(special_char_list, k=4)
    number_list = random.choices(num_list, k=4)
    password = ''
    for i in range(4):
        password += small_list[i]
        password += caps_list[i]
        password += spchar_list[i]
        password += number_list[i]
    print(password)

def user_gen_passwd(UI=None, *args):
    """Returns password generated by user input
    Accepts UI as a list or str, if UI is not
    declared returns a computer generated password"""
    if UI is None:
        return comp_gen_passwd()
    else:
        l1, l2 = [], []
        print()
        for i in range(len(args)):
            l1 = args[i]
            for j in l1:
                l2.append(j)
        for i in UI:
            l2.append(i)
        random.shuffle(l2)
        random.shuffle(l2)
        print(*l2)


def reverse_date(date):
    for i in ['/', '\'', '-', ' ', '|']:
        if i in date:
            spl = date.split(str(i))
            break
    x = ''
    for i in spl[::-1]:
        x += i + '-'
    x = x.rstrip('-')
    print(x)


def reverse_swap(sentence):
    """returns str(sentence) in reverse order and
    swaps lower-case to upper-case and vice versa"""
    sentence = str(sentence)
    sentence = sentence.split()
    s = ''
    for i in range(len(sentence)):
        s += sentence[-i-1] + ' '
    s = s.rstrip()
    print(s.swapcase())


def prime_factors(n):
    """
    :returns: prime factors of `n`
    """
    facs = []
    while n % 2 == 0:
        facs.append(2)
        n = n / 2
    for i in range(3, int(n**(1/2))+1, 2):
        while n % i == 0:
            facs.append(i)
            n = n / i
    if n > 2:
        facs.append(n)
    return facs


def fraction(n):
    """:param:   `n`: float
       :returns: fractional form of `n`
    """
    if int(str(n).split('.')[1]) == 0:
            return f"{str(n).split('.')[0]}/1"
    denom = str(10 ** len(str(n).split('.')[1]))
    numer = str(n).split('.')[0] + str(n).split('.')[1]
    num_facts = list(prime_factors(int(numer)))
    den_facts = list(prime_factors(int(denom)))
    num = 1
    den = 1
    for i in num_facts:
            if i in den_facts:
                    num_facts.remove(i)
                    den_facts.remove(i)
    for i in num_facts:
            num *= i
    for i in den_facts:
            den *= i
    return f"{int(num)}/{int(den)}"


def Lprint(string, interval=100):
    """prints `ivalchar` characters at a time
    of a very large string: `text` if `into` is None
    if `into` provided:
        inserts `text` into `into` tkinter Text object
        at index: `insertindex`.
    if `insertindex` not provided:
        inserts `text` at 'end'
    """
    start = 0
    limit = len(string)
    end = interval
    while True:
        if end > limit:
            end = limit
            print(' '.join(string[start:end]))
            break
        print(' '.join(string[start:end]))
        start += interval
        end += interval


def flattenNestedList(nestedList):
    """
    Converts a nested list to a flat list
    Example:    
    >>> d = [1, 2, 3, 4,
                [5, 6, 7, 8, 9], 10, 11, 12,
                [
                    [13, 14, 15, 16], 17, 18, 19
                ], 20
            ]
    >>> print(flattenNestedList(d))
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    """
    flatList = []
    for elem in nestedList:
        if isinstance(elem, list):
            flatList.extend(flattenNestedList(elem))
        else:
            flatList.append(elem)    
    return flatList
