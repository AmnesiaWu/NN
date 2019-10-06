import re
regex = re.compile(r'[1-9]\d{5}') #^abc表示以abc开头的字符串 ,abc$表示以abc结尾的字符串, [abc]表示a或b或c, abc*表示字符c从0开始向无穷大扩展, +表示从1开始，abc{n}表示扩展c n次
ans = re.findall(regex,'BIT 152634 456782')#search:匹配成功就行  match:从开头匹配,没有就说明不成功
print(ans)
temp = regex.split('BIT 152634 BBB456782', maxsplit=1)#maxsplit 表示最大分割数，剩下的以一个整体保存
print(temp)
temp_search = re.search(regex, 'BIT 152634 BBB456782')
print(temp_search.group())