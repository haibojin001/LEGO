def parse_value(self):
        current_char = self.current_char()

        if current_char == '{':
            return self.parse_object()
        elif current_char == '[':
            return self.parse_array()
        elif current_char == '"':
            return self.parse_string()
        elif current_char.isdigit() or current_char == '-':
            return self.parse_number()
        elif current_char.isalpha():
            return self.parse_boolean_or_null()
        else:
            raise ValueError(f"Unexpected character: {current_char}")