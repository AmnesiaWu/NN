# encoding:utf-8 #
class Solution:
    def bin_rev(self, int_input):
        """
        :param int_input: int
        :return:int
        """
        bin_input = bin(int_input)
        result_bin = bin_input[::-1][:-2] # 把输入的整数的二进制数翻转
        result_int = int(result_bin, 2)
        return result_int

if __name__ == '__main__':
    solution = Solution()
    int_input = eval(input("请输入一个整数:"))
    result = solution.bin_rev(int_input)
    print(result)