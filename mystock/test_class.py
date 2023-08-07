class MyClass:
    def __init__(self, value):
        self.value = value

    def run(self):
        print("value: " + str(self.value))


class MyClass2(MyClass):
    def __init__(self, value):
        super().__init__(value)
        self.value += 20


def create_instance(aaa):
    instance = aaa(10)
    instance.run()
    return


if __name__ == '__main__':
    create_instance(MyClass)
    create_instance(MyClass2)
