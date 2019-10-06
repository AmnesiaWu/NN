import re, collections

def get_all_words_dic(path):
    with open(path) as f:
        words = re.findall(r'[a-z]+', f.read().lower())
        words_dic = collections.defaultdict(lambda : 1)
        for i in words:
            words_dic[i] += 1
        return words_dic

def word_editor1(word):
    alph ='wqertyuioplkjhgfdsazxcvbnm'
    len_ = len(word)
    return set([word[0:i] + word[i + 1:] for i in range(len_)] + # 删除
               [word[0:i] + c + word[i:] for i in range(len_) for c in alph] + # 插入
               [word[0:i] + word[i + 1] + word[i] + word[i + 2:] for i in range(len_ - 1)] + # 交换
               [word[0:i] + c + word[i + 1:] for i in range(len_ + 1) for c in alph] # 替换
               )
def word_editor2(word):
    return set(s2 for s1 in word_editor1(word) for s2 in word_editor1(s1))

def known(words, al_words_dic):
    return set(word for word in words if word in al_words_dic)

def correct(word, al_words_dic):
    candidates = known([word], al_words_dic) or known(word_editor1(word), al_words_dic) or known(word_editor2(word), al_words_dic)
    return max(candidates, key=lambda x:al_words_dic[x])

def main():
    path = 'test.txt'
    al_words_dic = get_all_words_dic(path)
    result = correct('timm', al_words_dic)
    print(al_words_dic)

if __name__ == '__main__':
    main()

