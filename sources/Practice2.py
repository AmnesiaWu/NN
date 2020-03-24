# encoding:utf-8 #
# 动态规划算法
class Solution:
    def mini(self, triangle):
        """
        :type triangle: list[list[int]]
        :rtype: int
        """
        n = len(triangle)
        dp = triangle[-1]
        for i in range(n - 2, -1, -1):
            for j in range(i + 1):
                # 从倒数第二层向上
                dp[j] = triangle[i][j] + min(dp[j], dp[j + 1])
        return dp[0]