import tiktoken

class Tokenizier:
    def __init__(self):
        self.tokenizer = tiktoken.get_encoding("gpt2") # tokenizer
        self.vocab_list = {} # dictionary mapping words to an int number or id
        self.token_list = {} # same as dictionary but with the words encoded and mapped
        self.special_list = {} # hold all unique charcaters that arent letters or numbers 
    def is_english(self,text):
        try:
            text.encode(encoding='utf-8').decode('ascii')
        except UnicodeDecodeError:
            return False
        else:
            return True
    '''
    Functions for special charcaters (uneeded right now)
    def special_encoding(self, text):
        ##using byte pair encoding to add special charcaters(non stadard english) to the vocab_list
        self.vocab_list.update({text : len(self.vocab_list) + 1})
        self.token_list.update({text.encode() : len(self.token_list) + 1})
        return 
    def encode_and_add(self, text):
        ### split and store raw text in vocab
        text = text.split()
        text = sorted(set(text))
        ## returns and stores the key values only for comparison
        keys = self.vocab_list.keys()
        ## loops and adds and key not already in the diction into both the vocab(uncoded) and the token_list(coded)
        for i in range(len(text)):
            if text[i] not in keys:
                if not self.is_english(text[i]):
                    self.special_encoding(text[i])
                else:
                    self.vocab_list.update({text[i] : len(self.vocab_list) + 1})
                    ####Encode always returns a list do you must put [0] or the index to get to the actual value and no tthe list itself
                    self.token_list.update({self.tokenizer.encode(text[i])[0]: len(self.token_list) + 1})  
        new = list(self.token_list.keys())  
        return new 
    '''
    def decodes(self, text):
        final = []
        for item in text:
            if isinstance(item, int):
                t = self.tokenizer.decode([item])
                final.append(t)
            if isinstance(item, bytes):
                t = item.decode("utf-8")
                final.append(t)
        return final
    def encode(self,text):
        return self.tokenizer.encode(text)
    def get_vocab_length(self):
        return len(self.vocab_list)
    def get_token_list_length(self):
        return len(self.token_list)
'''
Testing 
import sys
import io
sys.stdout  = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
token = Tokenizier()
sample_text = "Thisw is a 테스트용 텍스트 bu"
sample_text_2 = " i hate everything here lmao"
print(token.encode_and_add(sample_text))      
print(token.decodes(token.encode_and_add(sample_text))) 

'''  


