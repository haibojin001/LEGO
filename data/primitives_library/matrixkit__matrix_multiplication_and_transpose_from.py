def __matmul__(self, ma):
        if not self.multipliable(ma):
            raise MatrixError("Matrixes are not multipliable")
        matrix = ma.transpose()
        m = []
        for i in range(self.rows):
            m.append([])
            for j in range(matrix.rows):
                m[i].append(
                    sum(
                        matrix.matrix[j][k] * self.matrix[i][k]
                        for k in range(self.cols)
                    )
                )
        return Matrix(m)