class Solution(object):
    def permute(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        def dfs(nums, used, curr, result):
            if len(curr) == len(nums):
                result.append(curr[:])
                return
            for i in range(len(nums)):
                if not used[i]:
                    curr.append(nums[i])
                    used[i] = True
                    dfs(nums, used, curr, result)
                    curr.pop()
                    used[i] = False
        curr = []
        result = []
        used = [False for i in nums]
        dfs(nums, used, curr, result)
        return result
    #Your input
    #[1,2,3]
    #Output
    #[[1,2,3],[1,3,2],[2,1,3],[2,3,1],[3,1,2],[3,2,1]]
    #Expected
    #[[1,2,3],[2,1,3],[2,3,1],[1,3,2],[3,1,2],[3,2,1]]