# MINED PRIMITIVE (Focus 1 co-occurrence pipeline)
# pid: python-datascience::pdg540::math.exp+math.sqrt
# name: math_primitive
# summary: Uses math.exp, math.sqrt across 2 repos
# anchor_symbols: ['math.exp', 'math.sqrt']
# observed in 2 repos: ['devAmoghS__Machine-Learning-with-Python', 'wooey__Wooey']...

# --- from devAmoghS__Machine-Learning-with-Python::helpers/probabilty.py::normal_pdf ---
def normal_pdf(x, mu=0, sigma=1.0):
    sqrt_two_pi = math.sqrt(2 * math.pi)
    return math.exp(-(x - mu) ** 2 / 2 / sigma ** 2) / (sqrt_two_pi * sigma)

# --- from wooey__Wooey::wooey/tests/scripts/gaussian.py::main ---
def main():
    args = parser.parse_args()
    u = args.mean
    s = abs(args.std)
    variance = s**2
    amplitude = 1 / (s * math.sqrt(2 * math.pi))
    fit = lambda x: [
        amplitude * math.exp((-1 * (xi - u) ** 2) / (2 * variance)) for xi in x
    ]
    # plot +- 4 standard deviations
    X = np.linspace(u - 4 * s, u + 4 * s, 100)
    Y = fit(X)
    plt.plot(X, Y)
    plt.title("Gaussian distribution with mu={0:.2f}, sigma={1:.2f}".format(u, s))
    plt.savefig("gaussian.png")
