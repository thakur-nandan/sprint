import string

STOPLIST = ["a", "about", "also", "am", "an", "and", "another", "any", "anyone", "are", "aren't", "as", "at", "be",
            "been", "being", "but", "by", "despite", "did", "didn't", "do", "does", "doesn't", "doing", "done", "don't",
            "each", "etc", "every", "everyone", "for", "from", "further", "had", "hadn't", "has", "hasn't", "have",
            "haven't", "having", "he", "he'd", "he'll", "her", "here", "here's", "hers", "herself", "he's",
            "him", "himself", "his", "however", "i", "i'd", "if", "i'll", "i'm", "in", "into", "is", "isn't", "it",
            "its", "it's", "itself", "i've", "just", "let's", "like", "lot", "may", "me", "might", "mightn't",
            "my", "myself", "no", "nor", "not", "of", "on", "onto", "or", "other", "ought", "oughtn't", "our", "ours",
            "ourselves", "out", "over", "shall", "shan't", "she", "she'd", "she'll", "she's", "since", "so", "some",
            "something", "such", "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then",
            "there", "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this", "those", "through",
            "tht", "to", "too", "usually", "very", "via", "was", "wasn't", "we", "we'd", "well", "we'll", "were",
            "we're", "weren't", "we've", "will", "with", "without", "won't", "would", "wouldn't", "yes", "yet", "you",
            "you'd", "you'll", "your", "you're", "yours", "yourself", "yourselves", "you've"]

printable = set('0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ ')
printableX = set('0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ-. ')
printable3X = set('0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ,.- ')

printableD = set('0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ.- ')
printable3D = set('0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ.- ')

STOPLIST_ = list(map(lambda s: ''.join(filter(lambda x: x in printable, s)), STOPLIST))

STOPLIST = {}
for w in STOPLIST_:
    STOPLIST[w] = True

def clean_passage(s, join=True):
    s = [(x.lower() if x in printable3X else ' ') for x in s]
    s = [(x if x in printableX else ' ' + x + ' ') for x in s]
    s = ''.join(s).split()
    s = [(w if '.' not in w else (' . ' if len(max(w.split('.'), key=len)) > 1 else '').join(w.split('.'))) for w in s]
    s = ' '.join(s).split()
    s = [(w if '-' not in w else w.replace('-', '') + ' ( ' + ' '.join(w.split('-')) + ' ) ') for w in s]
    s = ' '.join(s).split()
    # s = [w for w in s if w not in STOPLIST]

    return ' '.join(s) if join else s


def clean_query(s, join=True):
    s = [(x.lower() if x in printable3D else ' ') for x in s]
    s = [(x if x in printableD else ' ' + x + ' ') for x in s]
    s = ''.join(s).split()
    s = [(w if '.' not in w else (' ' if len(max(w.split('.'), key=len)) > 1 else '').join(w.split('.'))) for w in s]
    s = ' '.join(s).split()
    s = [(w if '-' not in w else (' ' if len(min(w.split('-'), key=len)) > 1 else '').join(w.split('-'))) for w in s]
    s = ' '.join(s).split()
    s = [w for w in s if w not in STOPLIST]

    return ' '.join(s) if join else s