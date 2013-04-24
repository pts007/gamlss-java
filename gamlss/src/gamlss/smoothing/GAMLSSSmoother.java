package gamlss.smoothing;

import gamlss.algorithm.AdditiveFit;

import org.apache.commons.math3.linear.ArrayRealVector;

/**
 * @author Dr. Vlasios Voudouris, Daniil Kiose, 
 * Prof. Mikis Stasinopoulos and Dr Robert Rigby.
 *
 */
public interface GAMLSSSmoother {
		
	/**
	 * The main fitting method, initiate appropriate smoothing 
	 * function according to incoming parameters.
	 * @param additiveFit -object of AdditiveFit class
	 * @return residuals
	 */
	ArrayRealVector solve(AdditiveFit additiveFit);
	
	/**
	 * Get variance.
	 * @return variance
	 */
  	ArrayRealVector getVar();
}
