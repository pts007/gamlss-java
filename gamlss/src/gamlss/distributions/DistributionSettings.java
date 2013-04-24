/*
  Copyright 2012 by Dr. Vlasios Voudouris and ABM Analytics Ltd
  Licensed under the Academic Free License version 3.0
  See the file "LICENSE" for more information
*/
package gamlss.distributions;
/**
 * 01/08/2012
 * @author Dr. Vlasios Voudouris, Daniil Kiose, Prof. Mikis Stasinopoulos and Dr Robert Rigby
 *
 */

public class DistributionSettings {

	// Supported distribution parameters
	/** mu = 1 */
	public final static int MU = 1;
	/** sigma = 2 */
	public final static int SIGMA = 2;
	/** nu = 3 */
	public final static int NU = 3;
	/** tau = 4 */
	public final static int TAU = 4;
	
	// Supported links 
	/** IDENTITY =0 */
	public final static int IDENTITY = 0;
	/** LOG =1 */
	public final static int LOG = 1;
	/** INVERSE=2 */
	public final static int INVERSE = 2;
	/** OWN=3 */
	public final static int OWN = 3;
	/** LOGSHIFTTO2=4 */
	public final static int LOGSHIFTTO2 = 4;
	
	
	// Supported Distribution types
	/** Continuous = 0 */
	public final static int CONTINUOUS = 0;
	/** Discrete = 1 */
	public final static int DISCRETE = 1;
	/** Mixed = 2 */
	public final static int MIXED = 2;
	
	// Supported Distributions
	/** Normal distribution = 0 */
	public final static int NO = 0;
	/** Student's t-distribution = 1 */
	public final static int TF = 1;
	/** Gamma distribution = 2*/
	public final static int GA = 2;
	/** Generalized t distribution = 3 */
	public final static int GT = 3;
	/** Skew t type3 = 4 */
	public final static int ST3 = 4;
	/** Skew t type4 = 5 */
	public final static int ST4 = 5;
	/** Johnson SU original = 6 */
	public final static int JSUo = 6;
	/** Student's t-distribution2 = 7 */
	public final static int TF2 = 7;
	/** Standardized skew t distribution = 8 */
	public final static int SST = 8;
	/** Box-Cox Power Exponential = 9 */
	public final static int BCPE = 9;
	/** Skew t (Azzalini type 1) = 10 */
	public final static int ST1 = 10;
	/** Power Exponential = 11*/
	public final static int PE = 11;
	
	// Supported fitting algorithms
	/** Rigby and Stasinopoulos algorithm  RS = 0 */
	public final static int RS = 1;
	/** Cole and Green fitting algorithm CG = 1 */
	public final static int CG = 2;
	/** Global Optimization algorithm GO = 2 */
	public final static int  GO = 3;
	/** Mixed algorithm, 20 loops of RS and 20 loops of CG */
	public final static int RS20CG20 = 0;
	
	// Supported smoothing methods
	public final static int ML = 0;
	
	public final static int ML1 = 1;
	
	public final static int EM = 2;
	
	public final static int GAIC = 3;
	
	public final static int GCV = 4;
	
	public static int DISTR = NO;
	
	public static int FITTING_ALG = RS;
	
	
	
}
