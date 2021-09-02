import math
class Vector:
    def __init__(self, val=None):
        if val:
            self.val = val
        else:
            self.val = ()


    def add(self, val):
        self.val += (val , )

    def __str__(self):
        return str(self.val)

    def __repr__(self):
        return self.__str__()


    def get_size(self):
        if len(self) == 0:
            return 0
        pow_sum = 0
        for v in self.val:
            pow_sum += v*v
        return math.sqrt(pow_sum)



    def div_by_num(self,num):
        ans = Vector()
        for v in self.val:
            ans.add(v/num)
        return ans


    def normalize(self):
        self.val = self.div_by_num(self.get_size())


    def add_vector(self, vector):
        ans = Vector()
        if self.get_size() == vector.get_size():
            for i in range(len(self.val)):
                ans.add(self.val[i] + vector.val[i])
        return ans


    def __len__(self):
        return len(self.val)


    @staticmethod
    def multi_2_vec(vector1, vector2):
        ans = 0
        for i in range(len(vector1.val)):
            ans += vector1.val[i] * vector2.val[i]
        return ans


    @staticmethod
    def similaroty_cos(vector1, vector2):
        if vector2.get_size() == 0 or vector1.get_size() == 0:
            return 0.0
        return Vector.multi_2_vec(vector1, vector2)/(vector1.get_size() * vector2.get_size())

