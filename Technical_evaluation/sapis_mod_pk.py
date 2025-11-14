
import requests
import pandas as pd
# import summaries_all
# from summaries_all import result as result_summaries

"""Calls on sapis and applies it to a text"""
def sapis_call(text):
    url = "https://sapis.it.liu.se/api"
    r = requests.post(url, data=text.encode('utf-8'))
    result = r.json()[0]
    #print(result)
    return result


"""Compares two dictionaries with identical keys and returns any keys with
     changed values along with those values and any parent keys"""


def compare_dicts(dict1, dict2, parent_keys=None):
    if parent_keys is None:
        parent_keys = []

    changed_keys = []
    
    for key in dict1.keys():
        if key not in dict2:
            print(f"Key '{key}' is missing in dict2 in the parent keys '{parent_keys + [key]}'")
            continue
        if isinstance(dict1[key], dict) and isinstance(dict2[key], dict):
            changed_keys.extend(compare_dicts(dict1[key], dict2[key], parent_keys=parent_keys + [key]))
        elif dict1[key] != dict2[key]:
            if key in ["input", "tokenized", "tagged", "parsed", "paragraphs", "stilett"]:
                #to see what stilett changes remove "stilett"
                continue
    
            separator = ""
            changed_keys.append([separator.join(parent_keys + [key]) + str(key), dict1[key], dict2[key]])
            #print(f"In '{parent_keys + [key]}' Key '{key}' has changed value from {dict1[key]} to {dict2[key]}", "\n")
        
    return changed_keys

# text1 = 'Natten till den 13 maj kan den som gillar att lyssna på fågelsång, men är för trött för att själv sitta på en stubbe i skogen och vaka när vårfåglarna vaknar, knäppa på radions P1. Möjligen kan man betrakta mitten på maj som startskottet för nattsångarna. Men var på plats redan vid midnatt för att fånga in de riktiga nattsångarna. Så fort gryningsljuset kommer, blandas deras nattliga sång upp med morgonserenader från alla håll av rödstjärtar, trastar, bofinkar, rödhakar. Var söker man de nattaktiva fåglarna, utöver ugglornas hoande ?. Där finns gott hopp om att höra surrande gräshoppsångare eller ute i vassen kärrsångare, rörsångare och trastsångare. En lite udda nattaktiv fågel är nattskärra, som får sökas i helt annan miljö. Man ska ha tur om man får syn på en väl kamouflerad nattskärra, som brukar jämföras med en brunmelerad barkbit där den ligger på stigen eller längs en grov trädgren.'
# text2 = 'Men var på plats redan vid midnatt för att fånga in de riktiga nattsångarna. Där finns gott hopp om att höra surrande gräshoppsångare eller ute i vassen kärrsångare, rörsångare och trastsångare. En lite udda nattaktiv fågel är nattskärra, som får sökas i helt annan miljö. Var söker man de nattaktiva fåglarna, utöver ugglornas hoande ?. Så fort gryningsljuset kommer, blandas deras nattliga sång upp med morgonserenader från alla håll av rödstjärtar, trastar, bofinkar, rödhakar. Natten till den 13 maj kan den som gillar att lyssna på fågelsång, men är för trött för att själv sitta på en stubbe i skogen och vaka när vårfåglarna vaknar, knäppa på radions P1. Möjligen kan man betrakta mitten på maj som startskottet för nattsångarna. Man ska ha tur om man får syn på en väl kamouflerad nattskärra, som brukar jämföras med en brunmelerad barkbit där den ligger på stigen eller längs en grov trädgren.'


if __name__ == "__main__":
    compare_dicts(sapis_call(text1), sapis_call(text2))
