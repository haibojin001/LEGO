def dual_step(self, z, y, x, mu):
        z = self.proj(x - np.dot(self.At, y), mu * self.rho)
        y = np.dot(self.dual_q, self.rho * (np.dot(self.A, x - z) - self.b))
        x = x - self.t * (np.dot(self.At, y) + z)
        return z, y, x