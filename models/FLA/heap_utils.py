import ctypes
from typing import Type
def create_heap_item_class(password_length:int)->Type[ctypes.Structure]:  
    """This function is needed to have a dynamic class system, 
       so that when the class is created we can allocate the appropriate amount of space for the password

    Args:
        password_length (int): the maximum length of the password, this is then used in to create the class 

    Raises:
        ValueError: This is raised if the passed password at instantiation time is too long.
        TypeError: This is raised when doing the operation < if the second object which is being compared to is of another type. 

    Returns:
        _type_: it returns a customized version of the class HeapItem
    """

    class HeapItem(ctypes.Structure):
        """
        This class is a representation of the contenents of the heap we are using.

        Inherits From:
         ctypes.Structure

        Raises:
            ValueError: if the password given at instanciation time is too big for the buffer
            TypeError: when doing a comparison operation with a value of a different type 
        """
        
        _fields_ = [
             ("prob", ctypes.c_double),
            ("password", ctypes.c_char * password_length) 
            #? Do we need to use utf-16 or utf-32 since python's str type is one or the other depending on implementation
           
        ]
        """
        C equivalent: 
            struct {
            char password [password_length]; password_length must be a variable since each time it could be different 
                                                 therefore using an appropriate amount of memory for the task at end
            float probability; 
            } HeapItem;
        """


        def __init__(self, probability:float, password:str):
            super().__init__()
            # Pad or truncate to exact length
            pwd_bytes:bytes = password.encode('utf-8')

            if len(pwd_bytes) > password_length: #prevent buffer overflow
                raise ValueError("The password is too long")

            self.password = pwd_bytes.ljust(password_length, b'\0')
            self.prob = probability

        #Comparison functions
        def __lt__(self, other)-> bool:
            if not isinstance(other,self.__class__):
                raise TypeError("HeapItem  cannot be less than an object of a different type")
            return self.prob < other.prob

        def __eq__(self, other)->bool:
            if not isinstance(other,self.__class__):
                raise TypeError("HeapItem cannot be equal with an object of a different type")
            return self.prob == other.prob and self.password == other.password
    
        def __le__(self,other)->bool:
            if not isinstance(other,self.__class__):
                raise TypeError("HeapItem cannot be less or equal than an object of a different type")
            return self.__lt__(other) or self.__eq__(other)
        
        def __gt__(self,other)->bool:
            if not isinstance(other,self.__class__):
                raise TypeError("HeapItem cannot be bigger than object of a different type")
            return not (self.__lt__(other)  or self.__eq__(other))
        
        def __ge__(self,other)->bool:
            if not isinstance(other,self.__class__):
                raise TypeError("HeapItem cannot be bigger or equal than object of a different type")
            return not self.__lt__(other) 

        def __sizeof__(self)->int:
            return ctypes.sizeof(self)        

        def __repr__(self)->str:
            return f"({self.prob},{self.password})"

        @property
        def password_string(self)->str:
        # self.password might be stored as bytes or string representation
            if isinstance(self.password, bytes):
                return self.password.rstrip(b'\0').decode('utf-8')
            elif isinstance(self.password, str):
                # If it's stored as string representation like "b'monichella'"
                if self.password.startswith("b'") and self.password.endswith("'"):
                    # Extract the actual content between b' and '
                    content = self.password[2:-1]
                    return content
                else:
                    return self.password.rstrip('\0')
            else:
                # It's a ctypes array, convert to bytes first
                return bytes(self.password).rstrip(b'\0').decode('utf-8')
   
    return HeapItem