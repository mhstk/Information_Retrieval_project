class Node:
    def __init__(self):
        self.data = None
        self.order = None
        self.next = None

class LinkedList:
    def __init__(self):
        self.head = None

    def addNode(self, data , order=-1):
        if order == -1:
            order = data
        curr = self.head
        if curr is None:
            n = Node()
            n.data = data
            n.order = order
            self.head = n
            return

        if curr.order > order:
            n = Node()
            n.data = data
            n.order = order
            n.next = curr
            self.head = n
            return

        while curr.next is not None:
            if curr.next.order > order:
                break
            curr = curr.next
        n = Node()
        n.data = data
        n.order = order
        n.next = curr.next
        curr.next = n
        return

    def __str__(self):
        data = []
        curr = self.head
        while curr is not None:
            data.append(curr.data)
            curr = curr.next
        return "[%s]" %(', '.join(str(i) for i in data))


    def to_list(self):
        data = []
        curr = self.head
        while curr is not None:
            data.append(curr.data)
            curr = curr.next
        return data

    def __repr__(self):
        return self.__str__()