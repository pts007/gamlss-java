/*
  Copyright 2012 by Dr. Vlasios Voudouris and ABM Analytics Ltd
  Licensed under the Academic Free License version 3.0
  See the file "LICENSE" for more information
  
  Dr. Vlasios Voudouris, Daniil Kiose, Prof. Mikis Stasinopoulos and Dr Robert Rigby
*/
package gamlss.distributions;


import org.apache.commons.math3.linear.ArrayRealVector;

/**
 * Interface defining a GAMLSS Family distribution with distribution parameters  mu, sigma, nu & tau.
 * </br>
 * </br>01/08/2012
 * @author Dr. Vlasios Voudouris, Daniil Kiose, Prof. Mikis Stasinopoulos and Dr Robert Rigby
 *
 */
 
 
public interface GAMLSSFamilyDistribution 
{
	/** Initializes the distribution parameters
	 * @param y - response variable */
	public void initialiseDistributionParameters (ArrayRealVector y);
	
	/** first derivative with respect to the "whatDistParameter" parameter
	 * @param whatDistParameter - distribution parameter
	 * @param y - response variable */
	public ArrayRealVector firstDerivative(int whatDistParameter, ArrayRealVector y);
	
	/** second derivative with respect to the "whatDistParameter" parameter
	 * @param whatDistParameter - distribution parameter
	 * @param y - response variable */
	public ArrayRealVector secondDerivative(int whatDistParameter, ArrayRealVector y );
	
	 /** Calculates second cross derivative
	 * @param whichDistParameter - distribution parameter
	 * @param y - response variable
	 * @return vector of second cross derivative values
	 */
	public ArrayRealVector secondCrossDerivative(int whichDistParameter1, int whichDistParameter2, ArrayRealVector y);
	
	/** increament of global deviance 
	 * 
	 * @param y - response variable 
	 * @return - global deviance increament vector
	 */
	public ArrayRealVector globalDevianceIncreament (ArrayRealVector y);

	/** check is the Y is valid for the speficied distribution
	 * @param y - response variable */
	public boolean isYvalid(ArrayRealVector y);
	
	/** checked if all distribution parametes are valid 
	 * @param whatDistParameter - distribution parameter */
	public boolean areDistributionParametersValid (int whichDistParameter);
	
	/** setts the values of distribution parameter (mu, sigma, nu or tau) vector 
	 *@param whichDistParameter - distribution parameter
	 * @param fvDistributionParameter - fitted values of Distribution Parameter*/
	public void setDistributionParameter(int whichDistParameter, ArrayRealVector fvDistributionParameter);
	
	/** get the number of distribution parameters (currently up-to 4 distribution parameters) */
	public int getNumberOfDistribtionParameters();
	
	/** get the type (e.g., continuous) of distribution */
	public int getTypeOfDistribution();
	
	/** get family (e.g., Normal) of distribution */
	public int getFamilyOfDistribution();
	
	/** returns the link function type of the current distribution parameter
	 * @param whichDistParameter - distribution parameter */
	public int getDistributionParameterLink(int whichDistParameter);
	
	/** returns the vector of distribution parameter (mu, sigma, nu or tau) values 
	 * @param whichDistParameter - distribution parameter */
	public ArrayRealVector getDistributionParameter(int whichDistParameter);
	
	
}
