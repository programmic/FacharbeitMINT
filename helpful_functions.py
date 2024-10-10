import colors as colors
from colorsys import hsv_to_rgb
from math import sqrt, acos, exp

clear = "\033c"

def readFile(pName: str, pNr: int, extension: str = ".txt") -> list[str]:
    """
    Reads a file ending with a number, usefull when multible similar files are in use, like different tasks or configs

    Args:
        pName (String): Path to the file location and file name
        pNr (int): Files number for iterating through similar files
        extension (String): Files file extension. Defaults to '.txt'
        removeEmpty (Boolean): Toggles wether empty lines are removed.Defaults to True

    Returns:
        Array: array of Strings, each String a line
    """
    dateiname = pName + str(pNr) + extension
    with open(dateiname, "r", encoding = "utf-8") as data:
        tmp = data.read().split("\n")
        ausgabe = []
        for i in tmp:
            if not i == "":
                ausgabe.append(i)
    return ausgabe

def lenformat( pInput: str | int, pDesiredLength: int, character: str = " ", place: str = "back" ) -> int:
    """
    Extends the length of a given string or integer to a specified length for prettier terminal output

    Args:
        pInput (string, int): The text that is to be formated
        pDesiredLength (int): Amount of characters the text should occupy
        character (str, optional): Characters used to fill blank space.\nDefaults to " ".
        place (str, optional): Defines wether characters should be placed in front or behind text.\n
            Accepts: "front", "back"\n
                Defaults to "back"

    Returns:
        String: String, formated to fit your needs
    """
    if place == "back":
        return str(str(pInput) + str(character * int(int(pDesiredLength) - len(str(pInput)))))
    elif place == "front":
        return str(character * int(int(pDesiredLength) - len(str(pInput)))) + str(str(pInput))
    
def clearTerminal() -> None:
    """
    clears the Terminal
    """
    print("\033c", end="")  # Clears Python Console Output

def makeMatrix(
        pX: int, 
        pY: int, 
        pZ:int =1
        ) -> list:
    """
    Easy way to quickly generate empty matrix
    Args:
        pX (int): matrix x dimension
        pY (int): matrix y dimension
        pY (int): matrix z dimension.\n
            Defaults to 1

    Returns:
        matrix (array): 2-Dimensional, empty data matrix
    """
    ret = []
    for i in range (pY):
        ret.append([])
        for j in range( pX ):
            ret[i].append([])
            if pZ > 1:
                for n in range(pZ):
                    ret[i][j].append([])  
    return ret

def HSVpercentToRGB(
        H: float, 
        saturation: float = 100, 
        value: float = 100
        ) -> tuple[ float, float, float ]:
    """
    Gibt den RGB-Wert basierend auf dem Prozentsatz durch den Hue-Wert zurück.
    Args:
        percentage (int): Ein Prozentsatz (0 bis 100), der angibt, wie weit man durch den Hue-Wert fortgeschritten ist.
    Returns:
        RBG (tupel): Ein Tupel (R, G, B) mit den RGB-Werten.
    """
    if not (0 <= H <= 100):
        raise ValueError("Percentage must be between 0 and 100")
    hue = (H / 100.0) * 360
    hue_normalized = hue / 360.0
    r, g, b = hsv_to_rgb(hue_normalized, saturation/100, value/100)
    
    return (float(r * 255), float(g * 255), float(b * 255))

def RGBtoKivyColorCode(color: tuple) -> tuple[ float, float, float ]:
    """
    | Converts a color from standart RGB color space to Kivy color space,
    | which is clamped between ```0-1``` instead of the normal ```0-25```

    Args:
        colorRGB  (tuple): Takes a ```0 - 255``` RBG Tupel ```(R, G, B)```
    Returns:
        colorKivy (tuple): returns same color value in Kivy color space
    """
    return( float(color[ 0 ] / 255 ), float( color[ 1 ] / 255 ), float(color[ 2 ] / 255 ) )

def normalizeVector(vector: tuple) -> tuple:
    """
    Normalizes a given *n*-dimensional vector\n
    .. math:: ∥V∥ = sqrt( v^(2/1) + v^(2/2) + ⋯ + v^(2/n) )

    Args:
        vector (tuple): *n*-dimensional vector

    Raises:
        ValueError: does not accept zero vector

    Returns:
        tuple: normalized vector
    """
    # Calculate the magnitude of the vector
    magnitude = sqrt(sum(v**2 for v in vector))
    
    # Avoid division by zero
    if magnitude == 0:
        raise ValueError("Cannot normalize a zero vector")
    
    # Divide each component by the magnitude
    return [v / magnitude for v in vector]

def intersectsLineVec(
        p1 :  tuple [ float, float ],
        p2 :  tuple [ float, float ],
        vec : tuple [ float, float ],
        dir : tuple [ float, float ]
        ) -> bool:
    pass




def dot2(vector1, vector2):
    """
    Calculate the dot product of two vectors.
    
    Parameters:
    vector1: list or tuple
    vector2: list or tuple
    
    Returns:
    The dot product of the two vectors.
    """
    return sum(x * y for x, y in zip(vector1, vector2))

def dot3(vector1, vector2):
    """
    Calculate the dot product of two 3D vectors.
    
    Parameters:
    vector1: list or tuple of 3 elements (x1, y1, z1)
    vector2: list or tuple of 3 elements (x2, y2, z2)
    
    Returns:
    The dot product of the two vectors.
    """
    if len(vector1) != 3 or len(vector2) != 3:
        raise ValueError("Both vectors must have exactly 3 elements.")
    
    return vector1[0] * vector2[0] + vector1[1] * vector2[1] + vector1[2] * vector2[2]

def mag3(vector):
    """
    Calculate the magnitude of a 3D vector.
    
    Parameters:
    vector: list or tuple of 3 elements (x, y, z)
    
    Returns:
    The magnitude of the vector.
    """
    return sqrt(vector[0]**2 + vector[1]**2 + vector[2]**2)

def mag2(vector):
    """
    Calculate the magnitude of a 2D vector.
    
    Parameters:
    vector: list or tuple of 3 elements (x, y)
    
    Returns:
    The magnitude of the vector.
    """
    return sqrt(vector[0]**2 + vector[1]**2)

def vec2angleRad(vector1, vector2):
    """
    Calculate the angle between two 2D vectors in radians.
    
    Parameters:
    vector1: list or tuple of 2 elements (x1, y1)
    vector2: list or tuple of 2 elements (x2, y2)
    
    Returns:
    The angle between the two vectors in radians.
    """
    dot_prod = dot2(vector1, vector2)
    magnitude_v1 = mag2(vector1)
    magnitude_v2 = mag2(vector2)
    
    # Calculate cosine of the angle using the dot product formula
    cos_angle = dot_prod / (magnitude_v1 * magnitude_v2)
    
    # To avoid floating point inaccuracies, ensure the value is in the range [-1, 1]
    cos_angle = max(min(cos_angle, 1), -1)
    
    # Calculate the angle in radians
    angle_radians = acos(cos_angle)
    
    return angle_radians

def vec3angleRad(vector1, vector2):
    """
    Calculate the angle between two 3D vectors in radians.
    
    Parameters:
    vector1: list or tuple of 3 elements (x1, y1, z1)
    vector2: list or tuple of 3 elements (x2, y2, z2)
    
    Returns:
    The angle between the two vectors in degrees.
    """
    dot_prod = dot3(vector1, vector2)
    magnitude_v1 = mag3(vector1)
    magnitude_v2 = mag3(vector2)
    
    # Calculate cosine of the angle using the dot product formula
    cos_angle = dot_prod / (magnitude_v1 * magnitude_v2)
    
    # To avoid floating point inaccuracies, ensure the value is in the range [-1, 1]
    cos_angle = max(min(cos_angle, 1), -1)
    
    # Calculate the angle in radians
    angle_radians = acos(cos_angle)
    
    return angle_radians


def scalar_vector_mult(scalar, vec):
    """Multiplies a scalar by a vector"""
    return [scalar * x for x in vec]

def vector_add(vec1, vec2):
    """Adds two vectors element-wise"""
    return [x + y for x, y in zip(vec1, vec2)]

def vector_subtract(vec1, vec2):
    """Subtracts vec2 from vec1 element-wise"""
    return [x - y for x, y in zip(vec1, vec2)]

def transpose(matrix):
    """Transposes a matrix (list of lists)"""
    return list(map(list, zip(*matrix)))



#### Activation functions

def sigmoid(z):
    """Sigmoid activation function"""
    return 1.0 / (1.0 + math.exp(-z))

def sigmoid_prime(z):
    """Derivative of sigmoid function"""
    return sigmoid(z) * (1 - sigmoid(z))

def relu(z):
    """ReLU activation function"""
    return max(0, z)

def relu_prime(z):
    """Derivative of ReLU function"""
    return 1 if z > 0 else 0

def leaky_relu(z, alpha=0.01):
    """Leaky ReLU activation function"""
    return z if z > 0 else alpha * z

def leaky_relu_prime(z, alpha=0.01):
    """Derivative of Leaky ReLU function"""
    return 1 if z > 0 else alpha

def tanh(z):
    """Tanh activation function"""
    return math.tanh(z)

def tanh_prime(z):
    """Derivative of tanh function"""
    return 1.0 - math.tanh(z) ** 2