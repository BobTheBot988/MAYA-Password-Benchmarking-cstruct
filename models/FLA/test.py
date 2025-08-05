class heapItem:
    __slots__ = ("password_byte","probability")
    def __init__(self,password:str,probability:float,max_length:int):
        enc = password.encode('utf8')
        if len(enc) > (max_length*2):
            raise ValueError("The password is too long")
        self.password_byte = enc.ljust((max_length*2),b'\x00')
        self.probability = probability

    @property
    def password(self)->str:
        return self.password_byte.rstrip(b'\x00').decode('utf8')

    def __lt__(self,other)->bool:
        return self.probability < other.probability
    def __repr__(self)->str:
        return f"heapItem({self.password!r},{self.probability})"
    def sizeof(self)->int:
        return (object.__sizeof__(self)+
                sys.getsizeof(self.password_byte)+
                sys.getsizeof(self.probability))
    
import sys
from heapq import heappush, heappop

# Tuple approach
heap_tuples = []

# Class with slots approach
heap_objects = []

heap_struct = []
import ctypes

def create_heap_item_class(password_length:int):
    class HeapItem(ctypes.Structure):
        _fields_ = [
            ("password", ctypes.c_char * (password_length*4)), # This is done for utf-8 compatibility 
            ("probability", ctypes.c_float)
        ]
        
        def __init__(self, password, probability):
            super().__init__()
            # Pad or truncate to exact length
            pwd_bytes:bytes = password.encode('utf-8')

            if len(pwd_bytes) > (password_length*4): #prevent buffer overflow
                raise ValueError("The password is too long")

            self.password = pwd_bytes.ljust(password_length, b'\0')
            self.probability = probability
        
        def __lt__(self, other)-> bool:
            if type(other) is not type(self):
                raise TypeError("Cannot compare an HeapItem with an object of a different type")
            return self.probability < other.probability
        def __sizeof__(self)->int:
            return  ctypes.sizeof(self)        
        @property
        def password_string(self)->str:
            return self.password.rstrip('\0').decode('utf-8')    
    return HeapItem


for i in range(100_000_000):
    heappush(heap_tuples, (f"password_{i}", float(i)))
    heappush(heap_objects, heapItem(f"password_{i}", float(i),10))
    HeapItem = create_heap_item_class(10)
    heappush(heap_struct, HeapItem(f"password_{i}", float(i)))

tup_size:int = heap_tuples.__sizeof__()
obj_size:int = heap_objects.__sizeof__()
struct_size:int = heap_struct.__sizeof__()
for i in range(100_000_000):
    obj_size += heap_objects.pop().sizeof()
    tup_size += heap_tuples.pop().__sizeof__()
    struct_size += heap_struct.pop().__sizeof__()
print(f"Tuple heap size ==> {tup_size:d}")
print(f"Object heap size ==> {obj_size:d}")
print(f"CStruct heap size ==> {struct_size:d}")
