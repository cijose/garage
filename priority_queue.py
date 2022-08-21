import random
import unittest


class PriorityQueue:
    def __init__(self, is_max=False):
        self.array_ = [None]
        self.is_max_ = is_max
        self.size_ = 0

    def size(self):
        return self.size_

    def push(self, item: int):
        self.size_ += 1
        if len(self.array_) > self.size_:
            self.array_[self.size_] = item
        else:
            self.array_.append(item)
        index = self.size_
        while index > 1:
            parent = int(index / 2)
            if (self.is_max_ and self.array_[parent] < self.array_[index]) or (
                not self.is_max_ and self.array_[parent] > self.array_[index]
            ):
                self.array_[parent], self.array_[index] = (
                    self.array_[index],
                    self.array_[parent],
                )
                index = parent
            else:
                break

    def top(self) -> int:
        if self.size_ > 0:
            return self.array_[1]
        else:
            raise Exception("Sorry, the priority queue is empty!")

    def heapify(self, index: int):
        next_index = index * 2
        while next_index <= self.size_:
            if next_index + 1 <= self.size_ and (
                (self.is_max_ and self.array_[next_index] < self.array_[next_index + 1])
                or (
                    not self.is_max_
                    and self.array_[next_index] > self.array_[next_index + 1]
                )
            ):
                next_index += 1
            if (self.is_max_ and self.array_[index] < self.array_[next_index]) or (
                not self.is_max_ and self.array_[index] > self.array_[next_index]
            ):
                self.array_[next_index], self.array_[index] = (
                    self.array_[index],
                    self.array_[next_index],
                )
                index = next_index
                next_index *= 2
            else:
                break

    def pop(self):
        if self.size_ > 0:
            top = self.array_[1]
            self.array_[1], self.array_[self.size_] = (
                self.array_[self.size_],
                self.array_[1],
            )
            self.size_ -= 1
            self.heapify(1)
            return top
        else:
            raise Exception("Sorry, the priority queue is empty!")


class TestPriorityQueue(unittest.TestCase):
    def test_min_priority_queue(self):
        min_priority_queue = PriorityQueue(is_max=False)
        min_priority_queue.push(3)
        min_priority_queue.push(4)
        min_priority_queue.push(1)
        min_priority_queue.push(5)
        self.assertEqual(min_priority_queue.pop(), 1)
        self.assertEqual(min_priority_queue.pop(), 3)
        min_priority_queue.push(2)
        self.assertEqual(min_priority_queue.pop(), 2)

    def test_max_priority_queue(self):
        max_priority_queue = PriorityQueue(is_max=True)
        max_priority_queue.push(3)
        max_priority_queue.push(4)
        max_priority_queue.push(1)
        max_priority_queue.push(5)
        self.assertEqual(max_priority_queue.pop(), 5)
        self.assertEqual(max_priority_queue.pop(), 4)
        max_priority_queue.push(7)
        self.assertEqual(max_priority_queue.pop(), 7)


if __name__ == "__main__":
    min_priority_queue = PriorityQueue()
    max_priority_queue = PriorityQueue(True)
    items = [i + 1 for i in range(40)]
    random.shuffle(items)
    for item in items:
        min_priority_queue.push(item)
        max_priority_queue.push(item)
    while min_priority_queue.size() and max_priority_queue.size():
        print(min_priority_queue.pop(), max_priority_queue.pop())
    unittest.main()
