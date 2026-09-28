from jmespath import ast
from jmespath import exceptions
from jmespath import lexer
from jmespath import visitor
from jmespath.compat import with_repr_method


class Parser(object):
    BINDING_POWER = {
        'eof': 0,
        'unquoted_identifier': 0,
        'quoted_identifier': 0,
        'literal': 0,
        'rbracket': 0,
        'rparen': 0,
        'comma': 0,
        'rbrace': 0,
        'number': 0,
        'current': 0,
        'expref': 0,
        'colon': 0,
        'pipe': 1,
        'or': 2,
        'and': 3,
        'eq': 5,
        'gt': 5,
        'lt': 5,
        'gte': 5,
        'lte': 5,
        'ne': 5,
        'flatten': 9,
        'star': 20,
        'filter': 21,
        'dot': 40,
        'not': 45,
        'lbrace': 50,
        'lbracket': 55,
        'lparen': 60,
    }

    _PROJECTION_STOP = 10
    _CACHE = {}
    _MAX_SIZE = 512

    def __init__(self, lookahead=2):
        self.tokenizer = None
        self._tokens = [None] * lookahead
        self._buffer_size = lookahead
        self._index = 0

    def parse(self, expression):
        try:
            return self._CACHE[expression]
        except KeyError:
            pass

        result = self._do_parse(expression)

        if len(self._CACHE) >= self._MAX_SIZE:
            try:
                del self._CACHE[next(iter(self._CACHE))]
            except (KeyError, StopIteration, RuntimeError):
                return result

        self._CACHE[expression] = result
        return result

    def _do_parse(self, expression):
        try:
            return self._parse(expression)
        except exceptions.LexerError as error:
            error.expression = expression
            raise
        except exceptions.IncompleteExpressionError as error:
            error.set_expression(expression)
            raise
        except exceptions.ParseError as error:
            error.expression = expression
            raise

    def _parse(self, expression):
        self.tokenizer = lexer.Lexer().tokenize(expression)
        self._tokens = list(self.tokenizer)
        self._index = 0

        parsed = self._expression(0)
        if self._current_token() != 'eof':
            token = self._lookahead_token(0)
            raise exceptions.ParseError(
                token['start'],
                token['value'],
                token['type'],
                'Unexpected token: %s' % token['value'],
            )
        return ParsedResult(expression, parsed)

    def _expression(self, binding_power=0):
        token = self._lookahead_token(0)
        self._advance()

        nud = getattr(
            self,
            '_token_nud_%s' % token['type'],
            self._error_nud_token,
        )
        left = nud(token)

        current = self._current_token()
        while binding_power < self.BINDING_POWER[current]:
            led = getattr(self, '_token_led_%s' % current, None)
            if led is None:
                self._error_led_token(self._lookahead_token(0))
            self._advance()
            left = led(left)
            current = self._current_token()

        return left

    def _token_nud_literal(self, token):
        return ast.literal(token['value'])

    def _token_nud_unquoted_identifier(self, token):
        return ast.field(token['value'])

    def _token_nud_quoted_identifier(self, token):
        result = ast.field(token['value'])
        if self._current_token() == 'lparen':
            token = self._lookahead_token(0)
            raise exceptions.ParseError(
                0,
                token['value'],
                token['type'],
                'Quoted identifier not allowed for function names.',
            )
        return result

    def _token_nud_star(self, token):
        left = ast.identity()
        if self._current_token() == 'rbracket':
            right = ast.identity()
        else:
            right = self._parse_projection_rhs(self.BINDING_POWER['star'])
        return ast.value_projection(left, right)

    def _token_nud_filter(self, token):
        return self._token_led_filter(ast.identity())

    def _token_nud_lbrace(self, token):
        return self._parse_multi_select_hash()

    def _token_nud_lparen(self, token):
        expression = self._expression()
        self._match('rparen')
        return expression

    def _token_nud_flatten(self, token):
        left = ast.flatten(ast.identity())
        right = self._parse_projection_rhs(self.BINDING_POWER['flatten'])
        return ast.projection(left, right)

    def _token_nud_not(self, token):
        return ast.not_expression(
            self._expression(self.BINDING_POWER['not'])
        )

    def _token_nud_lbracket(self, token):
        if self._current_token() in ('number', 'colon'):
            right = self._parse_index_expression()
            return self._project_if_slice(ast.identity(), right)

        if (self._current_token() == 'star' and
                self._lookahead(1) == 'rbracket'):
            self._advance()
            self._advance()
            right = self._parse_projection_rhs(
                self.BINDING_POWER['star']
            )
            return ast.projection(ast.identity(), right)

        return self._parse_multi_select_list()

    def _token_nud_current(self, token):
        return ast.current_node()

    def _token_nud_expref(self, token):
        return ast.expref(
            self._expression(self.BINDING_POWER['expref'])
        )

    def _token_led_pipe(self, left):
        right = self._expression(self.BINDING_POWER['pipe'])
        return ast.pipe(left, right)

    def _token_led_or(self, left):
        right = self._expression(self.BINDING_POWER['or'])
        return ast.or_expression(left, right)

    def _token_led_and(self, left):
        right = self._expression(self.BINDING_POWER['and'])
        return ast.and_expression(left, right)

    def _token_led_eq(self, left):
        return self._parse_comparator(left, 'eq')

    def _token_led_ne(self, left):
        return self._parse_comparator(left, 'ne')

    def _token_led_lt(self, left):
        return self._parse_comparator(left, 'lt')

    def _token_led_lte(self, left):
        return self._parse_comparator(left, 'lte')

    def _token_led_gt(self, left):
        return self._parse_comparator(left, 'gt')

    def _token_led_gte(self, left):
        return self._parse_comparator(left, 'gte')

    def _token_led_dot(self, left):
        if self._current_token() == 'star':
            self._advance()
            right = self._parse_projection_rhs(
                self.BINDING_POWER['star']
            )
            return ast.value_projection(left, right)

        if self._current_token() == 'lbracket':
            self._advance()
            right = self._parse_multi_select_list()
            return ast.subexpression(left, right)

        if self._current_token() == 'lbrace':
            self._advance()
            right = self._parse_multi_select_hash()
            return ast.subexpression(left, right)

        right = self._expression(self.BINDING_POWER['dot'])
        return ast.subexpression(left, right)

    def _token_led_lbracket(self, left):
        if self._current_token() in ('number', 'colon'):
            right = self._parse_index_expression()
            return self._project_if_slice(left, right)

        if (self._current_token() == 'star' and
                self._lookahead(1) == 'rbracket'):
            self._advance()
            self._advance()
            right = self._parse_projection_rhs(
                self.BINDING_POWER['star']
            )
            return ast.projection(left, right)

        right = self._parse_multi_select_list()
        return ast.subexpression(left, right)

    def _token_led_lbrace(self, left):
        right = self._parse_multi_select_hash()
        return ast.subexpression(left, right)

    def _token_led_flatten(self, left):
        flattened = ast.flatten(left)
        right = self._parse_projection_rhs(
            self.BINDING_POWER['flatten']
        )
        return ast.projection(flattened, right)

    def _token_led_filter(self, left):
        condition = self._expression(0)
        self._match('rbracket')
        right = self._parse_projection_rhs(
            self.BINDING_POWER['filter']
        )
        return ast.filter_projection(left, condition, right)

    def _token_led_lparen(self, left):
        if left['type'] != 'field':
            token = self._lookahead_token(-1)
            raise exceptions.ParseError(
                token['start'],
                token['value'],
                token['type'],
                'Invalid function name: %s' % token['value'],
            )
        return ast.function_expression(
            left['value'],
            self._parse_function_args(),
        )

    def _parse_function_args(self):
        arguments = []

        if self._current_token() == 'rparen':
            self._advance()
            return arguments

        while True:
            arguments.append(self._expression(0))
            if self._current_token() == 'comma':
                self._advance()
                continue
            self._match('rparen')
            return arguments

    def _parse_comparator(self, left, comparator):
        right = self._expression(self.BINDING_POWER[comparator])
        return ast.comparator(comparator, left, right)

    def _parse_index_expression(self):
        if (self._lookahead(0) == 'colon' or
                self._lookahead(1) == 'colon'):
            return self._parse_slice_expression()

        token = self._lookahead_token(0)
        result = ast.index(token['value'])
        self._advance()
        self._match('rbracket')
        return result

    def _parse_slice_expression(self):
        parts = [None, None, None]
        index = 0
        current = self._current_token()

        while current != 'rbracket' and index < 3:
            if current == 'colon':
                index += 1
                if index == 3:
                    self._raise_parse_error_for_token(
                        self._lookahead_token(0),
                        'syntax error',
                    )
                self._advance()
            elif current == 'number':
                parts[index] = self._lookahead_token(0)['value']
                self._advance()
            else:
                self._raise_parse_error_for_token(
                    self._lookahead_token(0),
                    'syntax error',
                )
            current = self._current_token()

        self._match('rbracket')
        return ast.slice(*parts)

    def _parse_projection_rhs(self, binding_power):
        if self.BINDING_POWER[self._current_token()] < self._PROJECTION_STOP:
            return ast.identity()
        return self._expression(binding_power)

    def _project_if_slice(self, left, right):
        if right['type'] == 'slice':
            return ast.projection(left, right)
        return ast.index_expression(left, right)

    def _parse_multi_select_list(self):
        expressions = []

        while True:
            expressions.append(self._expression(0))
            if self._current_token() == 'comma':
                self._advance()
                continue
            self._match('rbracket')
            break

        return ast.multi_select_list(expressions)

    def _parse_multi_select_hash(self):
        pairs = []

        while True:
            token = self._lookahead_token(0)
            if token['type'] not in (
                'unquoted_identifier',
                'quoted_identifier',
            ):
                self._raise_parse_error_for_token(
                    token,
                    'Expected an identifier',
                )

            self._advance()
            self._match('colon')
            value = self._expression(0)
            pairs.append(ast.key_val_pair(token['value'], value))

            if self._current_token() == 'comma':
                self._advance()
                continue

            self._match('rbrace')
            break

        return ast.multi_select_dict(pairs)

    def _match(self, token_type):
        if self._current_token() == token_type:
            self._advance()
            return

        token = self._lookahead_token(0)
        self._raise_parse_error_for_token(
            token,
            'Expecting: %s, got: %s' % (token_type, token['type']),
        )

    def _lookahead(self, number):
        return self._lookahead_token(number)['type']

    def _lookahead_token(self, number):
        return self._tokens[self._index + number]

    def _current_token(self):
        return self._lookahead(0)

    def _advance(self):
        self._index += 1

    def _error_nud_token(self, token):
        if token['type'] == 'eof':
            raise exceptions.IncompleteExpressionError(
                token['start'],
                token['value'],
                token['type'],
                'Invalid jmespath expression: Incomplete expression',
            )

        self._raise_parse_error_for_token(
            token,
            'Invalid token: %s' % token['value'],
        )

    def _error_led_token(self, token):
        self._raise_parse_error_for_token(
            token,
            'Unexpected token: %s' % token['value'],
        )

    def _raise_parse_error_for_token(self, token, message):
        raise exceptions.ParseError(
            token['start'],
            token['value'],
            token['type'],
            message,
        )


@with_repr_method
class ParsedResult(object):
    def __init__(self, expression, parsed):
        self.expression = expression
        self.parsed = parsed

    def search(self, value, options=None):
        interpreter = visitor.TreeInterpreter(options)
        return interpreter.visit(self.parsed, value)

    def __repr__(self):
        return 'ParsedResult(expression=%r, parsed=%r)' % (
            self.expression,
            self.parsed,
        )