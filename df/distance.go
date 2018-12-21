package df

import "math"

//Cosine calcute cosine similarity
//
// $$sim(d_i ,d_j ) = \frac{ ∑_k w_{ki} · w_{kj}  }{ \sqrt{∑_k (w_{ki})^2  }  \sqrt{∑_k (w_{kj})^2  }   } $$
//	return similarity value.
// 采用map主要是为了应对解决稀疏问题
func Cosine(s map[string]float64, t map[string]float64) float64 {

	if len(s) < 1 || len(t) < 1 {
		return .0
	}

	numerator, sL, tL, temp2 := .0, .0, .0, .0

	for key, value := range s {
		temp1 := value
		if v, ok := t[key]; !ok {
			temp2 = 0
		} else {
			temp2 = v
		}
		delete(t, key)
		numerator += temp1 * temp2
		sL += temp1 * temp1
		tL += temp2 * temp2
	}

	for _, value := range t {
		tL += value * value
	}

	result := numerator / (math.Sqrt(sL * tL))
	return result
}
