def lcs(sequence1, sequence2):
    m = len(sequence1)
    n = len(sequence2)

    # Create DP table
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    # Fill DP table
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if sequence1[i - 1] == sequence2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    # Backtrack to find LCS
    i, j = m, n
    lcs_sequence = []

    while i > 0 and j > 0:
        if sequence1[i - 1] == sequence2[j - 1]:
            lcs_sequence.append(sequence1[i - 1])
            i -= 1
            j -= 1
        elif dp[i - 1][j] > dp[i][j - 1]:
            i -= 1
        else:
            j -= 1

    lcs_sequence.reverse()

    return "".join(lcs_sequence), dp[m][n]