import json
import os


class Language:

    def __init__(self):
        self.lang = None

    def load(self, path: str):
        if not os.path.exists(path):
            print("File does not exist...")

        f = open(path, "r")
        self.lang = json.loads(f.read())
        f.close()

    def words(self):
        wordDict = {}

        for w in self.lang['words']:
            wordDict[w["word"]] = w["phonemes"]

        return wordDict

    def word_phonemes(self, word):
        words = self.words()
        return words[word]

    def get(self, lang: str, case: str, word):
        if self.lang is None:
            return ""

        if "response" in self.lang:
            return self.lang["response"][lang]["default"][case] % word
        else:
            return ""

    def get_special(self, lang: str):
        if self.lang is None:
            return ""

        if "response" in self.lang:
            return self.lang["response"][lang]["special"]
        else:
            return []

    def get_special_feedback(self, lang):
        special_cases = []
        if self.lang is None:
            return special_cases

        if "response" in self.lang:
            for case in self.lang["response"][lang]["special"]:
                special_cases.append(case["feedback"])
            return special_cases


