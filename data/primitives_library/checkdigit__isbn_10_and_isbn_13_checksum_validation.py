def validate_isbn10(isbn: str) -> bool:
    """
    Validates an ISBN-10 using the check digit algorithm.
    
    Args:
        isbn: ISBN-10 string (hyphens/spaces will be removed)
        
    Returns:
        True if valid ISBN-10, False otherwise
    """
    # Remove hyphens and spaces
    cleaned = re.sub(r'[-\s]', '', isbn).upper()
    
    # Must be exactly 10 characters
    if len(cleaned) != 10:
        return False
    
    # First 9 must be digits, 10th can be digit or X
    if not cleaned[:9].isdigit():
        return False
    if not (cleaned[9].isdigit() or cleaned[9] == 'X'):
        return False
    
    # Calculate checksum
    total = 0
    for i in range(9):
        total += int(cleaned[i]) * (10 - i)
    
    # Add check digit
    if cleaned[9] == 'X':
        total += 10
    else:
        total += int(cleaned[9])
    
    return total % 11 == 0