def Mult1(self, m, p1, p2):
    # Ensure valid range
    p1 = max(0, min(1, p1))
    p2 = max(0, min(1, p2))
    
    # Ensure p1 + p2 <= 1
    if p1 + p2 > 1:
        # Scale down proportionally
        scale = 1 / (p1 + p2)
        p1 *= scale
        p2 *= scale
    
    p3 = max(0, 1 - p1 - p2)
    
    # Ensure probabilities sum to 1
    probs = [p1, p2, p3]
    sum_probs = sum(probs)
    if sum_probs > 0:
        normalized_probs = [p/sum_probs for p in probs]
    else:
        normalized_probs = [1/3, 1/3, 1/3]  # Equal fallback
    
    return np.random.multinomial(m, normalized_probs)[0]