def levenshteinDistanceDP (token1, token2):
    
    # first, create a 2D matrix of (m+1, n+1)
    distances = np.zeros((len(token1) + 1, len(token2) + 1))

    #in numpy 2d arrays are [c][r] opposite to java
    # init the first row (0) to equal 0 1 2 3 etc
    # the first row corresponds to the first token
    for t1 in range(len(token1) + 1):
        distances[t1][0] = t1

    #init the first column to equal 0 1 2 3
    # the first column corresponds to the second token
    for t2 in range(len(token2) + 1):
        distances[0][t2] = t2

    
    # at this point thr matrix would look like this
    #
    # token1 = foo
    # token2 = bar
    #
    #    | "" | f  | o  | o  |
    # "" | 0  | 1  | 2  | 3  |  
    #  b | 1  |    |    |    |
    #  a | 2  |    |    |    |
    #  r | 3  |    |    |    |
    #
    #
    #  these values are derived from the number of operations it would
    #  take each substring to match an empty string ""
    #  operations can only be insert, replace or delete
    #
    #  0: subtrings "" and "" match, so no operation is taken 
    #  1: substrings "f" and "b" require 1 deletion each (f => "" and b => "")
    #  2: substrings "fo" and "ba" require 2 deletions each  ("fo" => "" and "ba" =>)
    #  3: substrings "foo" and "bar" require 3 deletions each ("foo" => "" and "bar" =>)


    # time to calc distances between all substrings 
    
    # iterate through each cell
    for t1 in range(1, len(token1) + 1): # starts at 1 because column 0 filled
        for t2 in range(1, len(token2) + 1): # starts at 1 because row 0 filled

        # now we compare 

        # token1 = bat and token2 = bum
        #
        #       ""  b 
        #  "" | 0 | 1 |
        #  b  | 1 | * |

        # in cell * of the matrix we are comparing 2 substrings "b" and "b"
        # in this situation, the last character of both substrings is 'b' (first character is "")
        # when the last character is equal, the distance is the same as the top left corner of
        # its immeadiate 2x2 neighborhood => "b" and "b" don't need any action taken, just like 
        # "" and "" don't need any action taken  
        # 
        # therefore the distance at * = 0

            if (token1[t1-1] == token2[t2-1]): # compare the characters 
                distances[t1][t2] = distances[t1 - 1][t2 - 1] # get the top left corner


        # token1 = bat and token2 = bum
        #
        #       "" |  b |  a  
        #  "" | 0  |  1 |  2 |
        #  b  | 1  |  0 |  * |

        # in cell * of the matrix we are comparing 2 substrings "ba" and "b"
        # in this situation, the last character of both substrings is NOT equal. a != b
        # the distance in cell * is found by finding the minimum of its 3 neighboring costs
        # in its 2x2 neighborhood and adding a cost of 1 

        # therefore the distance at * = 1

            else:
                a = distances[t1][t2 - 1] # find value of cell above current 
                b = distances[t1 - 1][t2] # find value of cell next to current
                c = distances[t1 - 1][t2 - 1] # find top left cell value
                
                if (a <= b and a <= c): # a is minimum
                    distances[t1][t2] = a + 1
                elif (b <= a and b <= c): # b is minimum 
                    distances[t1][t2] = b + 1
                else: # c is minimum
                    distances[t1][t2] = c + 1 

    # let's print the matrix 
    printDistances(distances, len(token1), len(token2))
    
    # to find the minmum distance, you return the bottom right most value in the matrix
    # this would be the value for last char in token1 and token 2
    return distances[len(token1)][len(token2)]