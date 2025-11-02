#=========================================================================
# PWQ QR CODE GENERATOR
#=========================================================================
#==================== MODULE IMPORTING =============================#
import cv2
import numpy as np
#=========================================================================
# ================== GENERATOR CLASS =================================#
class PWQ__QR_CODE_GENERATOR:
    def __init__(self, Version_Size, Box_Size, Input_Link, mask_ID):
        # ============ GENERAL ============================================#
        self.Version_Size_Decoder = {  # Version size decoder
            # | Format -> "version size" : (size in px + 2, total data bits 'L', 'M', 'Q', 'H', total data+EC bytes, remainder bits)
            "1": (23, 19, 16, 13, 9, 26, 0),
            "2": (27, 34, 28, 22, 16, 44, 7),
            "3": (31, 55, 44, 34, 26, 70, 7),
            "4": (35, 70, 56, 44, 34, 100, 7),
            "5": (39, 100, 80, 64, 48, 134, 7),
            "6": (43, 134, 108, 86, 62, 172, 7),
            "7": (47, 172, 132, 100, 76, 196, 0),
            "8": (51, 196, 160, 122, 88, 242, 0),
            "9": (55, 240, 192, 140, 110, 292, 0),
            "10": (59, 280, 224, 160, 130, 346, 0),
            "11": (63, 324, 264, 192, 150, 404, 0),
            "12": (67, 370, 308, 224, 176, 466, 0),
            "13": (71, 428, 352, 260, 198, 532, 0),
            "14": (75, 461, 384, 288, 224, 581, 3),
            "15": (79, 523, 432, 320, 252, 655, 3),
            "16": (83, 589, 480, 360, 282, 733, 3),
            "17": (87, 647, 528, 390, 310, 815, 3),
            "18": (91, 721, 600, 432, 338, 901, 3),
            "19": (95, 795, 672, 480, 382, 991, 3),
            "20": (99, 861, 744, 528, 403, 1085, 3),
            "21": (103, 932, 816, 572, 439, 1156, 4),
            "22": (107, 1006, 909, 618, 461, 1258, 4),
            "23": (111, 1094, 970, 672, 511, 1364, 4),
            "24": (115, 1174, 1035, 720, 535, 1474, 4),
            "25": (119, 1276, 1134, 768, 593, 1588, 4),
            "26": (123, 1370, 1248, 816, 625, 1706, 4),
            "27": (127, 1468, 1326, 909, 658, 1828, 4),
            "28": (131, 1531, 1451, 970, 698, 1921, 3),
            "29": (135, 1631, 1542, 1035, 742, 2051, 3),
            "30": (139, 1735, 1637, 1094, 790, 2185, 3),
            "31": (143, 1843, 1732, 1172, 842, 2323, 3),
            "32": (147, 1955, 1839, 1263, 898, 2465, 3),
            "33": (151, 2071, 1994, 1322, 958, 2611, 3),
            "34": (155, 2191, 2113, 1409, 983, 2761, 3),
            "35": (159, 2306, 2238, 1477, 1051, 2876, 0),
            "36": (163, 2434, 2369, 1563, 1093, 3034, 0),
            "37": (167, 2566, 2506, 1631, 1139, 3196, 0),
            "38": (171, 2702, 2632, 1732, 1219, 3362, 0),
            "39": (175, 2812, 2780, 1839, 1273, 3532, 0),
            "40": (179, 2956, 2894, 1990, 1367, 3706, 0)
        }

        self.Version_Size = int(self.Version_Size_Decoder[Version_Size][0]) # Decode the version size in px
        self.Box_Size = Box_Size # Set the box size to self attribute
        self.Input_Link = Input_Link # Set the link to self attribute
        self.Binary_Link_List = [] # Create empty list for link bits in binary
        self.QR_Code_Frame = np.full((self.Version_Size * self.Box_Size + 0 * self.Box_Size, self.Version_Size * self.Box_Size  + 0 * self.Box_Size, 3),
                                     (255, 255, 255),
                                     dtype=np.uint8)  # Create an array sized according to version and box size
        # ============ QR CODE PREPARE ==================================#
        # ============ BIT ENCODING =====================================#
        self.Binary_String_Segments = [] # Create an empty list for the bit segments
        self.Binary_Bit_Pixel_Values_Black = [] # Create an empty list of all black pixels
        self.Binary_Bit_Pixel_Values_White = [] # Create an empty list of all black pixels
        # ============ ERROR CORRECTION BITS ==========================#
        self.Error_Correction_Bits_Overlay = [
            *((8, y1) for y1 in range(0, 9)),  # TOP LEFT
            *((x1, 8) for x1 in range(0, 9)),
            *((self.Version_Size - 10 + x1, 8) for x1 in range(0, 8)), # BOTTOM LEFT
            *((8, self.Version_Size - 10 + y1) for y1 in range(0, 8)),
        ]
        self.Integer_Codewords_Values = [] # Create an empty list for codewords values in base 10
        self.Primitive_Polynomial = 0x11d # Primitive polynomial hex value for GF(256) decoder generation
        self.Mask_ID = mask_ID
        self.BCH_Encoding_Mask = '101010000010010'
        # =========== PATTERN CREATION =================================#
        self.Finder_Pattern_Cords_White = [
            # ------------------- TOP LEFT FINDER PATTERN ---------------------#
            # WHITE #
            *((x1, 1) for x1 in range(1, 6)),
            *((x1, 5) for x1 in range(1, 6)),
            *((1, y1) for y1 in range(1, 6)),
            *((5, y1) for y1 in range(1, 6)),
            # ------------------ TOP RIGHT FINDER PATTERN --------------------#
            # WHITE #
            *((self.Version_Size - 3 - x1, 1) for x1 in range(1, 6)),
            *((self.Version_Size - 3 - x1, 5) for x1 in range(1, 6)),
            *((self.Version_Size - 4, y1) for y1 in range(1, 5)),
            *((self.Version_Size - 8, y1) for y1 in range(1, 5)),
            # --------------- BOTTOM LEFT FINDER PATTERN ------------------#
            # WHITE #
            *((x1, self.Version_Size - 4) for x1 in range(1, 6)),
            *((x1, self.Version_Size - 8) for x1 in range(1, 6)),
            *((5, self.Version_Size - 4 - y1) for y1 in range(1, 5)),
            *((1, self.Version_Size - 4 - y1) for y1 in range(1, 5)),
            # ------------------- ALIGNMENT PATTERN ---------------------#
            *((self.Version_Size - 10 + x, self.Version_Size - 10) for x in range(0, 3)),
            *((self.Version_Size - 10 + x, self.Version_Size - 8) for x in range(0, 3)),
            *((self.Version_Size - 10, self.Version_Size - 10 + y) for y in range(0, 3)),
            *((self.Version_Size - 8, self.Version_Size - 10 + y) for y in range(0, 3)),
            # ((self.Version_Size - 9, self.Version_Size - 9)),
            # ------------------- TIMING PATTERNS ---------------------#
            *((6, self.Version_Size - 10 - 2 * y) for y in range(0, (self.Version_Size - 14 - 1) // 2)),
            *((7 + 2 * y, 6) for y in range(0, (self.Version_Size - 14 - 1) // 2)),
            # ---------------------- SEPARATOR ---------------------------#
            *((7, y1) for y1 in range(0, 8)),  # TOP LEFT
            *((x1, 7) for x1 in range(0, 8)),
            *((self.Version_Size - 10, y1) for y1 in range(0, 8)),  # TOP RIGHT
            *((self.Version_Size - 10 + x1, 7) for x1 in range(0, 8)),
            *((7, self.Version_Size - 10 + y1) for y1 in range(0, 8)),  # BOTTOM LEFT
            *((x1, self.Version_Size - 10) for x1 in range(0, 8)),
        ]

        self.Finder_Pattern_Cords_Black = [
            # --------- SPECIFIC SINGLE-POINT COORDINATES ----------------#
            (0, 0),  # 0, 0 -> Always Black
            (8, self.Version_Size - 10),  # 8, VS - 10 -> Alignment pattern's pixel

            # ------------------- TOP LEFT FINDER PATTERN ---------------------#
            # BLACK #
            *((0, y1) for y1 in range(1, 7)),
            *((x1, 0) for x1 in range(1, 7)),
            *((6, y1) for y1 in range(1, 7)),
            *((x1, 6) for x1 in range(1, 7)),
            *((x2 + 1, y2 + 1) for x2 in range(1, 4) for y2 in range(1, 4)),

            # ------------------ TOP RIGHT FINDER PATTERN --------------------#
            *((self.Version_Size - 3, y1) for y1 in range(0, 7)),
            *((self.Version_Size - x1 - 3, 0) for x1 in range(1, 7)),
            *((self.Version_Size - 9, y1) for y1 in range(1, 7)),
            *((self.Version_Size - 3 - x1, 6) for x1 in range(1, 7)),
            *((self.Version_Size - 9 + x2 + 1, y2 + 1) for x2 in range(1, 4) for y2 in range(1, 4)),

            # --------------- BOTTOM LEFT FINDER PATTERN ------------------#
            *((x, self.Version_Size - 3) for x in range(0, 7)),
            *((0, self.Version_Size - 3 - y) for y in range(1, 7)),
            *((x, self.Version_Size - 9) for x in range(0, 7)),
            *((6, self.Version_Size - 3 - y) for y in range(1, 7)),
            *((x2 + 1, self.Version_Size - 9 + y2 + 1) for x2 in range(1, 4) for y2 in range(1, 4)),

            # ------------------- ALIGNMENT PATTERN ---------------------#
            *((self.Version_Size - 11 + x, self.Version_Size - 11) for x in range(0, 5)),
            *((self.Version_Size - 11, self.Version_Size - 11 + y) for y in range(0, 5)),
            *((self.Version_Size - 11 + x, self.Version_Size - 7) for x in range(0, 5)),
            *((self.Version_Size - 7, self.Version_Size - 11 + y) for y in range(0, 5)),
            ((self.Version_Size - 9, self.Version_Size - 9)),

            # ------------------- TIMING PATTERNS ---------------------#
            *((6, self.Version_Size - 11 - 2 * y) for y in range(0, (self.Version_Size - 14 - 1) // 2)),
            *((8 + 2 * y, 6) for y in range(0, (self.Version_Size - 14 - 1) // 2))

        ]
    # ============ MASKING LOGIC ========================================================================================================================#
    def Masking_QR_Function(self, x, y, mask_ID):
        mask_ID = int(mask_ID, 2)
        if mask_ID == 0:
            return (x + y) % 2 == 0
        if mask_ID == 1:
            return y % 2 == 0
        if mask_ID == 2:
            return x % 3 == 0
        if mask_ID == 3:
            return (x + y) % 3 == 0
        if mask_ID == 4:
            return (x // 3 + y // 2) % 2 == 0
        if mask_ID == 5:
            return (x * y) % 2 + (x * y) % 3 == 0
        if mask_ID == 6:
            return ( (x * y) % 2 + (x * y) % 3 ) % 2 == 0
        if mask_ID == 7:
            return ( (x + y) % 2 + (x * y) % 3 ) % 2 == 0
        return False
    # ============ GLOBAL PATTERN ITERATOR MODULE ==================================================================================================#
    def Global_Pattern_Iterator_Module(self, Binary_String):
        x, y = self.Version_Size - 3, self.Version_Size - 3 # Compute the initial pixel values (bottom right corner)
        trend = "up" # Start with an upward trend
        Binary_String_Segments = []
        def Bit_Placement(bit, px, py):
            if bit == '1': # If pixel is black
                self.Binary_Bit_Pixel_Values_Black.append((px, py)) # Append it to the black pixels list
            else: # Else if its white
                self.Binary_Bit_Pixel_Values_White.append((px, py)) # Append it to the white pixels list
            #print(f"{bit} → ({px}, {py})") # Print the binary color of the pixel, and its 2D position

        self.Binary_String_Segments = [char for char in Binary_String]


        print("after", self.Binary_String_Segments)
        i = 0 # Set initial index value of bits to 0
        px_placed = 0 # Set value of total bits placed to 0
        final_column_mode = False
        Occupied_Pixel_Conjuction = set(self.Finder_Pattern_Cords_White) | set(self.Finder_Pattern_Cords_Black) | set(self.Error_Correction_Bits_Overlay) # Compute one conjunction of all lists, with occupied pixel positions
        while i < len(self.Binary_String_Segments): # Loop until the last index of the string has been reached
            coord1 = (x, y) #  Allocate coordinate to the right-hand side of the column
            coord2 = (x - 1, y)  # Allocate coordinate to the left-hand side of the column
            if x == 0 and y == 8:
               break
                # =================== CHECK TO SKIP OCCUPIED PIXELS ==========#
            if coord1 in Occupied_Pixel_Conjuction and coord2 in Occupied_Pixel_Conjuction: # Check if both coordinates are in the occupied conjunction
                if trend == "up": # If so, and the trend is 'up'
                    y -= 1 # Decrement y-value by 1 (move 1 pixel up)
                    if (y >= 8 and x > self.Version_Size - 10) or y == 0: # If either (y is greater than or equal to 8 AND x is greater than version size in px - 10) OR (y is 0)  ||| Check if we have hit the top limit
                        y = 0 # Set y to 0
                        trend = "down" # Set the trend to 'down'
                else: # Else if the trend is 'down'
                    y += 1 # Increment y-value by 1 (move 1 pixel down)
                    if y > self.Version_Size - 3: # If y is greater than version size in px - 3 ||| check if we have hit the bottom limit
                        y = self.Version_Size - 3 # If so set y to the limit
                        if not final_column_mode:
                            if x > 1:
                                x -= 2
                            else:
                                x = 0
                                final_column_mode = True
                        trend = "up" # Set the trend to 'up'
                #print(f"Cordinates {coord1} amd {coord2} found inside of ocupied space! Skipping current iteration") # Print the skipped coordinates
                continue # Continue with the next iteration

            # ======== INTENT PLACING THE RIGHT-SIDE PIXEL ===========#
            if coord1 not in Occupied_Pixel_Conjuction and i < len(self.Binary_String_Segments): # If the right-hand side coordinate is NOT in the place of any occupied pixel and the index is still in range of the string
                bit1 = self.Binary_String_Segments[i] # Compute the bit value at current index
                if self.Masking_QR_Function(*coord1, self.Mask_ID):
                    bit1 = '1' if bit1 == '0' else '0'
                Bit_Placement(bit1, *coord1) # Place the bit in its corresponding position
                Occupied_Pixel_Conjuction.add(coord1) # Add that pixel to the occupied pixels list
                i += 1 # Increment i to the next index
                px_placed += 1 # Increment pixels placed

            # ======== INTENT PLACING THE LEFT-SIDE PIXEL =============#
            if x != 0 and coord2 not in Occupied_Pixel_Conjuction and i < len(self.Binary_String_Segments): # If the left-hand side coordinate is NOT in the place of any occupied pixel and the index is still in range of the string
                bit2 = self.Binary_String_Segments[i] # Compute the bit value at current index
                Bit_Placement(bit2, *coord2) # Place the bit in its corresponding position
                Occupied_Pixel_Conjuction.add(coord2) # Add that pixel to the occupied pixels list
                i += 1 # Increment i to the next index
                px_placed += 1 # Increment pixels placed
            # ======== CHECK FOR TREND MOVEMENT =======================#
            if trend == "up": # If the trend is 'up'
                y -= 1 # ALWAYS decrement y (move 1 pixel up)
                if y < 0: # If the top limit (0) has been hit

                    y = 0  # Set y to the top limit (0)
                    if not final_column_mode:
                        if x > 1:
                            x -= 2
                        else:
                            x = 0
                            final_column_mode = True
                    trend = "down" # Set the trend to 'down'
                    continue # Continue with the next iteration
            else:  # Else if trend is 'down'
                y += 1 # ALWAYS Increment y (move 1 pixel down)
                bottom_limit = self.Version_Size - 3 # Set bottom limit to version size in px - 3
                if y > bottom_limit: # If the bottom limit (version size - 3) has been hit
                    y = bottom_limit # Set y to the bottom limit (version size - 3)
                    if not final_column_mode:
                        if x > 1:
                            x -= 2
                        else:
                            x = 0
                            final_column_mode = True

                    trend = "up" ## Set the trend to 'up'
                    continue # Continue with the next iteration

        print(f"✅ Done. Total bits placed: {i}") # Print total bits placed


    # ------------- CONVERT LINK INTO BINARY USING ASCII -------------------------------------------------------------------------------------------------------#
    def Convert_Integer_Into_Binary(self, Integer):
        return format(Integer, '08b')  # 8-bit binary

    def Convert_Link_Into_Binary(self):
        for char in self.Input_Link: # Loop thru all characters in the link
            binary_char = format(ord(char), "08b")  # Convert to 8-bit binary
            self.Binary_Link_List.append(binary_char) # Append the charater to the list
        return self.Binary_Link_List # Return Binary-converted link

    # --------- ERROR CORRECTION BITS ---------------------------------------------------------------------------------------------------------------------------------#
    def Convert_Binary_String_Into_8Bit_Integer_Values(self, string):
        for x in range(0, len(string), 8): # Loop thru the total length of the string
            current_8bit_chunk = string[x:x+8] # Compute an 8-bit binary chunk every iteration
            int_value = int(current_8bit_chunk, 2) # Convert that 8-bit chunk into 0-255 integer value
            self.Integer_Codewords_Values.append(int_value) # Append the int value to a list with all values
        return self.Integer_Codewords_Values # Return the list

    def Complete_Polynomial_w_EC_Zeros(self, cur_int_list, total_EC_bits):
        self.Complete_Polynomial = cur_int_list + [0] * total_EC_bits # Complete the polynomial by adding zeros for every EC bit
        n = len(self.Complete_Polynomial)  # Set n equal to the degree of polynomial (length of data codewords)
        self.Polynomial_Degree_Index_Value = [n - i for i in range(1, n)]  # Compute a list with matching exponents to the int codewords values list
        print("Complete Data Polynomial:", self.Complete_Polynomial) # Print the list
        return self.Complete_Polynomial, self.Polynomial_Degree_Index_Value # Return the complete polynomial list and the degree index list

    def Galois_Field_Addition(self, a, b):
        a_bin, b_bin = format(a, '08b'), format(b, '08b') # Convert both numbers into binary
        result_bin = '' # Precompute the result
        for bit_a, bit_b in zip(a_bin, b_bin): # Loop thru every bit of the number
            bit_c = 1 if bit_a != bit_b else 0 # if the bits are NOT equal set bit c to 1 else 0. ||| Manual XOR Gate
            result_bin += str(bit_c) # Add the bit c to the total result string
        return int(result_bin, 2) # Return integer-converted result of the addition

    def Galois_Field_Multiplication(self, a, b):
        if a == 0 or b == 0: # If BOTH numbers are 0
            return 0 # Retutn result equal to 0 and finish iteration
        result = self.Antilog_Table[((self.Log_Table[a] + self.Log_Table[b])) % 255] # Else, convert both numbers into their corresponding Log Table values, add the looked up values together, take modulo of 8-bit max, and look up result in the Antilog Table to get the final result value
        return result # Return the result

    def Galois_Field_Polynomial_Multiplication(self, polynomial_a, polynomial_b):
        result = [0] * (len(polynomial_a) + len(polynomial_b) - 1) # Pre-allocate the result's size
        for a in range(len(polynomial_a)): # Loop thru the length of polynomial a
            for b in range(len(polynomial_b)): # Loop thru the length of polynomial b
                product = self.Galois_Field_Multiplication(polynomial_a[a], polynomial_b[b]) # Compute GF(256) multiplication on the value at index a of polynomial a and value at index b of polynomial b
                result[a+b] = self.Galois_Field_Addition(result[a+b], product) # Add the value at index a + b of the result list with the product
        return result # Return the final result

    def Definition_Generator_Polynomial(self, EC_Bits):
        Generator = [1] # Create a generator with a starting 1-degree polynomial of 1
        for EC_bit in range(EC_Bits): # Loop thru the length of EC Bits
            Current_Generator_Term = [1, self.Antilog_Table[EC_bit]] # Compute the generator term (x + alpha^i) where x is 1
            Generator = self.Galois_Field_Polynomial_Multiplication(Generator, Current_Generator_Term)  # Compute the Polynomial Multiplication on the previous generator polynomial and the current generator term
        print("Generator Polynomial:",Generator) # Print the final generator polynomial
        return Generator # Return the final generator polynomial

    def Generate_Error_Correction_Bits(self, Data_Poly, Generator_Poly, byte_conversion=True,slicing_offset=0):
        if hasattr(Data_Poly, 'list'):
            Data = Data_Poly.copy() # Make a separate copy of the Data Complete Polynomial
        else:
            Data = Data_Poly
        for d in range(len(Data_Poly) - len(Generator_Poly) + 1): # Loop thru the diffrence of lengths between the generator and the data polynomial (Essentially loop thru the length of EC bits since Data poly is padded with zeros for every EC bit)
            coefficient = Data[d] # Assign the current coefficient at the current EC bit
            if coefficient != 0:  # If the coefficient is NOT 0
                for g in range(len(Generator_Poly)): # Loop thru the length of the generator
                    Data[d + g] = self.Galois_Field_Addition(                                         # 1. At the index of d + g of Data list
                        Data[d+g],                                                                                  # 2. Add the value at index d + g of Data list
                        self.Galois_Field_Multiplication(coefficient, Generator_Poly[g])   # 3. With the product of the current coefficient and current value of Generator Polynomial at index g
                    )
        remainder_bits = Data[-slicing_offset:] # Extract remainder bits by using slicing (-len(Gen): -> Extracts every value beyond the length of the generator => That's where the remainder bits are stored)
        print("EC Remainder Bits:",remainder_bits) # Print the Generated EC Bits
        if byte_conversion:
            remainder_binary_bit_list = [self.Convert_Integer_Into_Binary(bit) for bit in remainder_bits]
            return remainder_binary_bit_list  # Return the Generated EC Bits
        else:
            return remainder_bits


    def Generate_Galois_Field_Decoders(self):
        """
        Bitwise Operators in Python:
        & - AND Gate
        | - OR Gate
        ^ - XOR Gate
        ~ - NOT Gate
        << - Left Shift
        >> - Right Shift
        """
        Log_Table = [0] * 256 # Define log table as a list with 256 zeros
        Antilog_Table = [0] * 512 # Define antilog table as a list with double 256 zeros ||| double -> Overflow-safe indexing
        Primitive_Element_Alpha = 1 # Primitive element - 2 ||| Since GF(256) ranges from 0 to a byte of data, it's been defined to use the element = 2 because of ease of usage. ||| Divide by 2 because we use bit shifting instead of standard exponent multiplication
        Initial_Power = 1 # Set initial exponent to 1
        for i in range(255): # Loop thru the range of Galois Field 256
            Antilog_Table[i] = Initial_Power # Antilog table at index i is equal to current exponent
            Log_Table[Initial_Power] = i # Log table at index of current exponent is equal to i
            Initial_Power <<= Primitive_Element_Alpha # Each apply bit-shift to the exponent
            if Initial_Power & 0x100: # If it overflew yonder beyond 8-bit value (8-bit -> 0x100 in hex)
                Initial_Power ^= self.Primitive_Polynomial # Reduce module primitive polynomial by raising the exponent to the primitive polynomial
        for i in range(255, 512): # Loop thru duplicates in antilog table
            Antilog_Table[i] = Antilog_Table[i-255] # The final antilog log from 255-512 range, is equal to a duplicate between range 0-255
        print(f"Log table has been generated as: {Log_Table}")
        print(f"Antilog table has been generated as: {Antilog_Table}")
        self.Log_Table, self.Antilog_Table = Log_Table, Antilog_Table
        return self.Log_Table, self.Antilog_Table # Return both tables

    def BCH_Encoding_Information_Bits(self, format_data_bits):
        padded_bits = format_data_bits + '0' * 10
        format_data_list = [int(b) for b in padded_bits]
        BCH_Generator = [int(b) for b in '10100110111']
        remainder_bits = self.Generate_Error_Correction_Bits(format_data_list, BCH_Generator, byte_conversion=False, slicing_offset=10)
        remainder_str = ''.join(str(b) for b in remainder_bits)
        complete_format_data = format_data_bits + remainder_str
        self.final_format_bits = ''
        for a, b in zip(complete_format_data, self.BCH_Encoding_Mask):
            c = 1 if a != b else 0
            self.final_format_bits += str(c)
        print("Final Format EC Bits:", self.final_format_bits)
        return self.final_format_bits

    def Compute_Error_Correction_Bits(self):
        # Format info placement (15 bits total)
        format_coords = [
            # Top-left vertical (6 bits + 1 extra at (8,7))
            (0, 8), (1, 8), (2, 8), (3, 8), (4, 8), (5, 8), (7, 8), (8, 8),
            (8, 7), (8, 5), (8, 4), (8, 3), (8, 2), (8, 1), (8, 0),
            # Top-left horizontal (8 bits, skipping timing at (6,8))
        ]
        format_coords_duplicate = [
            (8, self.Version_Size - 3), (8, self.Version_Size - 4),
            (8, self.Version_Size - 5), (8, self.Version_Size - 6),
            (8, self.Version_Size - 7), (8, self.Version_Size - 8),
            (8, self.Version_Size - 9),
            # Top-right horizontal (8 bits)
            (self.Version_Size - 10, 8), (self.Version_Size - 9, 8),
            (self.Version_Size - 8, 8), (self.Version_Size - 7, 8),
            (self.Version_Size - 6, 8), (self.Version_Size - 5, 8),
            (self.Version_Size - 4, 8), (self.Version_Size - 3, 8),
        ]
        for i, (x_cell, y_cell) in enumerate(format_coords):
            x = x_cell * self.Box_Size + self.Box_Size
            y = y_cell * self.Box_Size + self.Box_Size
            print(self.final_format_bits[i], i, x_cell, y_cell)
            color = (0, 0, 0) if self.final_format_bits[i] == '1' else (255, 255, 255)
            cv2.rectangle(self.QR_Code_Frame, (x, y), (x + self.Box_Size, y + self.Box_Size), color, -1)

        for i, (x_cell, y_cell) in enumerate(format_coords_duplicate):
            x = x_cell * self.Box_Size + self.Box_Size
            y = y_cell * self.Box_Size + self.Box_Size
            color = (0, 0, 0) if self.final_format_bits[i] == '1' else (255, 255, 255)
            cv2.rectangle(self.QR_Code_Frame, (x, y), (x + self.Box_Size, y + self.Box_Size), color, -1)

    # --------------- INFORMATION PATTERNS (For version <7) --------------------------------------------------------------------------------------------------------#
    def Create_Information_Pattern(self):
        pass

    def Create_Window_Visualization(self):
        self.All_Pixel_Values_List_Black = self.Binary_Bit_Pixel_Values_Black + self.Finder_Pattern_Cords_Black
        self.All_Pixel_Values_List_White = self.Binary_Bit_Pixel_Values_White + self.Finder_Pattern_Cords_White
        # ------------------- VISUALISE EVERY BLACK PIXEL ---------------------#
        for idx, val in enumerate(self.All_Pixel_Values_List_Black): # Loop thru the pixel list
            x = val[0] * self.Box_Size + self.Box_Size # Current pixel's x -value
            y = val[1] * self.Box_Size + self.Box_Size # Current pixel's y -value

            cv2.rectangle(self.QR_Code_Frame, (x, y), (x + self.Box_Size, y + self.Box_Size), (0, 0, 0), -1) # Draw a rectangular pixel according to box size
        # ------------------- VISUALISE EVERY WHITE PIXEL ---------------------#
        for idx, val in enumerate(self.All_Pixel_Values_List_White):  # Loop thru the pixel list
            x = val[0] * self.Box_Size + self.Box_Size  # Current pixel's x -value
            y = val[1] * self.Box_Size + self.Box_Size  # Current pixel's y -value

            cv2.rectangle(self.QR_Code_Frame, (x, y), (x + self.Box_Size, y + self.Box_Size), (255, 255, 255),-1)  # Draw a rectangular pixel according to box size
        # ------------------ WINDOW VISUALIZATION ------------------------#
        cv2.imshow('QR Code', self.QR_Code_Frame) # Create a window from the array above, named 'QR Code'
        cv2.waitKey(0) #
        cv2.destroyAllWindows() # After user closes window -> delete that window
# ============== USER INPUT ==========================================#
Version_Size = "3" # Version size 1 - 40
Box_Size = 15 # Size of the QR code pixel
Error_Correction_Level = 'L' # Error Correction Level
Input_Link = "https://www.google.com" # Link
mask_ID = '110'
#============== COMPUTATION =========================================#
EC_Index = {'L': '01', 'M': '00', 'Q': '11', 'H': '10'}.get(Error_Correction_Level) # Decode the error correction level into an index
PWQ__QR_Code_Generation = PWQ__QR_CODE_GENERATOR(Version_Size, Box_Size, Input_Link, mask_ID) # Class call
Total_Data_Codewords = PWQ__QR_Code_Generation.Version_Size_Decoder[Version_Size][int(EC_Index, 2)] # Total codewords (data bits)
Total_Data_ED_Codewords = PWQ__QR_Code_Generation.Version_Size_Decoder[Version_Size][5]
Total_EC_Codewords = Total_Data_ED_Codewords - Total_Data_Codewords

Data_Format = "0100" # Numeric Format - 0001 | Alphanumeric - 0010 | Binary - 0100 | 1000 - Japanese Kanji
Link_Length = PWQ__QR_Code_Generation.Convert_Integer_Into_Binary(len(Input_Link)) # Convert the length of the link into binary
Binary_Link = PWQ__QR_Code_Generation.Convert_Link_Into_Binary()  # Convert link into binary


Info_Binary_String = Data_Format + Link_Length # Add the info bits (12 total, or 1.5 bytes)
# ============= REPEATING PATTERN COMPUTATION ===================#
base_binary = Info_Binary_String + ''.join(str(bit) for bit in Binary_Link) + '0000'  # Total amount of data bits without the repeating pattern

while len(base_binary) % 8 != 0: # Ensure rare case of unfilled byte of zeros
    print("Byte padding needed!")
    base_binary += '0' # Fill in the byte with zeros

current_bytes = len(base_binary) // 8 # Compute total bytes used so far
missing_bytes = Total_Data_Codewords - current_bytes # Compute total 'missing' bytes for the repeating pattern to fill in
if missing_bytes < 0: # Check if the the range of the link is aproporiate for the level of error correction and version size
    raise ValueError(f"Data too large for version/level: need {current_bytes} bytes but only {Total_Data_Codewords} available") # If not, then raise a ValueError

padding_bytes = ['11101100', '00010001'] # Allocate the repeating pattern
for i in range(missing_bytes): # Loop thru all missing bytes
    base_binary += padding_bytes[i % 2] # For each one, fill in with the repeating pattern correspondingly

assert len(base_binary) == Total_Data_Codewords * 8, "Final bit length mismatch" # Assert the total amount of bytes is equal to total amount of bits * 8

Complete_Binary_Encoded_String = base_binary # Assign the complete string of bytes

print(Complete_Binary_Encoded_String) # Print for debug
#----------- ERROR CORRECTION BITS ----------------------------------------#
PWQ__QR_Code_Generation.Generate_Galois_Field_Decoders() # Generate the Log Table and Antilog Table Decoders
Complete_Integer_Values = PWQ__QR_Code_Generation.Convert_Binary_String_Into_8Bit_Integer_Values(Complete_Binary_Encoded_String) # Convert the string binary data bits into 0-255 integer list
Complete_Data_Polynomial, Degree_Data_List = PWQ__QR_Code_Generation.Complete_Polynomial_w_EC_Zeros(Complete_Integer_Values, Total_EC_Codewords) # Complete the polynomial list with EC zero-padding
Generator_Polynomial = PWQ__QR_Code_Generation.Definition_Generator_Polynomial(Total_EC_Codewords) # Generate the Generator Polynomial for every EC bit
EC_Bits_Binary = PWQ__QR_Code_Generation.Generate_Error_Correction_Bits(Complete_Data_Polynomial, Generator_Polynomial, slicing_offset=Total_EC_Codewords) # Generate the Error Correction Bits
Complete_Binary_Encoded_String += ''.join(str(bit) for bit in EC_Bits_Binary)
Complete_Binary_Encoded_String += ''.join(str(bit) for bit in EC_Bits_Binary)
Remainder_Bits_Count = PWQ__QR_Code_Generation.Version_Size_Decoder[Version_Size][6]
print(f"Adding {Remainder_Bits_Count} remainder bits.")
Complete_Binary_Encoded_String += '0' * Remainder_Bits_Count
#----------- BCH FORMATTING + FORMAT STRIP BITS ----------------------#
Format_Strips_Data = EC_Index + mask_ID
BCH_Encoded_Format_Data = PWQ__QR_Code_Generation.BCH_Encoding_Information_Bits(Format_Strips_Data)
# ======== FUNCTION CALLS ============================================#
PWQ__QR_Code_Generation.Global_Pattern_Iterator_Module(Complete_Binary_Encoded_String) # Zig zag pattern executor
PWQ__QR_Code_Generation.Compute_Error_Correction_Bits() # Computes error correction bytes
PWQ__QR_Code_Generation.Create_Window_Visualization() # Visualises the pixels on a window
