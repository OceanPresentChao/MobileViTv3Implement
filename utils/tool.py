
import difflib
def diffDictKeys(dict1:dict,dict2:dict):
    d = difflib.Differ()
    diff = d.compare(getDictKeys(dict1).splitlines(), getDictKeys(dict2).splitlines(),)
    print ('\n'.join(list(diff)))


def getDictKeys(dict):
    res = ""
    for key in dict:
        res += key + "\n"
    return res