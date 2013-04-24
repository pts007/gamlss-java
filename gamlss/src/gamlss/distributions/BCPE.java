/*
  Copyright 2012 by Dr. Vlasios Voudouris and ABM Analytics Ltd
  Licensed under the Academic Free License version 3.0
  See the file "LICENSE" for more information
*/
package gamlss.distributions;

import gamlss.utilities.Controls;
import gamlss.utilities.GammaDistr;
import gamlss.utilities.MakeLinkFunction;

import java.util.Hashtable;
import org.apache.commons.math3.distribution.UniformRealDistribution;
import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.special.Gamma;
import org.apache.commons.math3.stat.descriptive.moment.Mean;
import org.apache.commons.math3.util.FastMath;

/**
 * @author Dr. Vlasios Voudouris, Daniil Kiose, 
 * Prof. Mikis Stasinopoulos and Dr Robert Rigby.
 */
public class BCPE implements GAMLSSFamilyDistribution {
	/** Number of distribution parameters. */
	private final int numDistPar = 4;
	/** Hashtable to hold vectors of distribution
	 *  parameters (mu, sigma, ...). */
	private Hashtable<Integer, ArrayRealVector> distributionParameters
								   = new Hashtable<Integer, ArrayRealVector>();
	/** Hashtable to hold types of link functions
	 *  for the distribution parameters. */
	private Hashtable<Integer, Integer> distributionParameterLink 
										   = new Hashtable<Integer, Integer>();
	/** vector of values of mu distribution parameter. */
	private ArrayRealVector muV;
	/** vector of values of sigma distribution parameter. */
	private ArrayRealVector sigmaV;
	/** vector of values of nu distribution parameter. */
	private ArrayRealVector nuV;
	/** vector of values of tau distribution parameter. */
	private ArrayRealVector tauV;
	/** Temporary vector for interim operations. */
	private ArrayRealVector tempV;
	/** Temporary int for interim operations. */
	private int size;
	/** Temporary array for interim operations. */
	private double[] c;
	/** Temporary array for interim operations. */
	private double[] logC;
	/** Temporary array for interim operations. */
	private double[] dlogcDt;
	/** Temporary array for interim operations. */
	private double[] z;
	/** Array of first derrivative values dl/dmu. */
	private double[] dldm;
	/** Array of first derrivative values dl/dsigma. */
	private double[] dlds;
	/** Array of first derrivative values dl/dnu. */
	private double[] dldn;
	/** Array of first derrivative values dl/dtau. */
	private double[] dldt;
	/** Object of the Gamma distribution .*/
	private GammaDistr gammaDistr;
	
	
//	private double h;
//	private double loglik;
///	private double s;
//	private double fS;

	/** This is the Box-Cox Power Exponential  distribution with default
	 * link (muLink="identity",sigmaLink="log", 
	 * nuLink="identity", tauLink="log"). */
	public BCPE() {
	
	this(DistributionSettings.IDENTITY, DistributionSettings.LOG, 
			DistributionSettings.IDENTITY, DistributionSettings.LOG);
	}
	
	/** 
	 * This is the Box-Cox Power Exponential distribution with supplied
	 *  link function for each of the distribution parameters.
	 * @param muLink - link function for mu distribution parameter
	 * @param sigmaLink - link function for sigma distribution parameter
	 * @param nuLink - link function for nu distribution parameter
	 * @param tauLink - link function for tau distribution parameter
	 */
	public BCPE(final int muLink, 
			   final int sigmaLink, 
			   final int nuLink, 
			   final int tauLink) {
	
		distributionParameterLink.put(DistributionSettings.MU, 	   
				MakeLinkFunction.checkLink(DistributionSettings.BCPE, muLink));
		distributionParameterLink.put(DistributionSettings.SIGMA,  
				MakeLinkFunction.checkLink(DistributionSettings.BCPE, sigmaLink));
		distributionParameterLink.put(DistributionSettings.NU,     
				MakeLinkFunction.checkLink(DistributionSettings.BCPE, nuLink));
		distributionParameterLink.put(DistributionSettings.TAU,    
				MakeLinkFunction.checkLink(DistributionSettings.BCPE, tauLink));
		
		gammaDistr = new GammaDistr();

	}

	/** Initialises the distribution parameters.
	 * @param y - response variable */
	public final void initialiseDistributionParameters(
												final ArrayRealVector y) {

		distributionParameters.put(DistributionSettings.MU, 
				setMuInitial(y));
		distributionParameters.put(DistributionSettings.SIGMA, 
				setSigmaInitial(y));
		distributionParameters.put(DistributionSettings.NU, 
				setNuInitial(y));
		distributionParameters.put(DistributionSettings.TAU,
				setTauInitial(y));
	}
	
	/**  Calculates initial value of mu, by assumption these 
	 * values lie between observed data and the trend line.
	 * @param y - vector of values of response variable
	 * @return  a vector of initial values of mu
	 */
	private ArrayRealVector setMuInitial(final ArrayRealVector y) {
		//mu.initial =  expression(mu <- (y+mean(y))/2)
		size = y.getDimension();
		double[] out = new double[size];
		Mean mean = new Mean();	
		double yMean = mean.evaluate(y.getDataRef());	
		for (int i = 0; i < size; i++) {
			out[i] = (y.getEntry(i) + yMean) / 2;
		}
		return new ArrayRealVector(out, false);
	}
	
	/** Calculates initial value of sigma.
	 * @param y - vector of values of response variable
	 * @return - a vector of  initial values of sigma
	 */
	private ArrayRealVector setSigmaInitial(final ArrayRealVector y) {
		//sigma.initial = expression(sigma<- rep(0.1, length(y)))
		tempV = new ArrayRealVector(y.getDimension());
		tempV.set(0.1);
		return tempV;
	}

	/** Calculates initial value of nu.
	 * @param y - vector of values of response variable
	 * @return - a vector of  initial values of nu
	 */
	private ArrayRealVector setNuInitial(final ArrayRealVector y) {	
		//nu.initial = expression(nu <- rep(1, length(y))),
		tempV = new ArrayRealVector(y.getDimension());
		tempV.set(1);
		return tempV;
	}

	/** Calculates initial value of tau.
	 * @param y - vector of values of response variable
	 * @return - a vector of  initial values of tau
	 */
	private ArrayRealVector setTauInitial(final ArrayRealVector y) {
		// tau.initial = expression(tau <-rep(2, length(y))),
		tempV = new ArrayRealVector(y.getDimension());
		tempV.set(2);
		return tempV;
	}

	/** Calculates a first derivative of the likelihood 
	 * function in respect to supplied distribution parameter.
	 * @param whichDistParameter - distribution parameter
	 * @param y - vector of values of likelihood function
	 * @return vector of first derivative of the likelihood 
	 */
	public final ArrayRealVector firstDerivative(final int whichDistParameter, 
	   	     									 final ArrayRealVector  y) {
		setInterimArrays(y, whichDistParameter);
		tempV = null;		
		switch (whichDistParameter) {
	    case DistributionSettings.MU:
	    	tempV = dldm(y);
	       break;
	    case DistributionSettings.SIGMA:
	    	tempV = dlds(y);
	      break;
	    case DistributionSettings.NU:
	    	tempV = dldn(y);
	      break;
	    case DistributionSettings.TAU:
	    	tempV = dldt(y);
	      break;
		default: 
			System.err.println("Requested first order "
					+ "derivative does not exist");
			break;
	    }			
		return tempV;
	}
	
	/**
	 * Set z, logC, c arrays.
	 * @param y - response variable
	 */
	private void setInterimArrays(final ArrayRealVector y, 
								  final int whichDistParameter) {
		muV     = distributionParameters.get(DistributionSettings.MU);
		sigmaV  = distributionParameters.get(DistributionSettings.SIGMA);
	 	nuV     = distributionParameters.get(DistributionSettings.NU);
	 	tauV    = distributionParameters.get(DistributionSettings.TAU);
		
	 	size = y.getDimension();
		z = new double[size];
		logC = new double[size];
		c = new double[size];
	 	for (int i = 0; i < size; i++) {
	 		//z <- ifelse(nu != 0,(((y/mu)^nu-1)/(nu*sigma)),log(y/mu)/sigma)
	 		if (nuV.getEntry(i) != 0) {
	 			z[i] = (FastMath.pow(y.getEntry(i) 
	 					       / muV.getEntry(i), nuV.getEntry(i)) - 1) 
	 								/ (nuV.getEntry(i) * sigmaV.getEntry(i));
	 		} else {
	 			
	 			z[i] = FastMath.log(y.getEntry(i) / muV.getEntry(i))
	 													/ sigmaV.getEntry(i);
	 		}
	 		
	 		//log.c <- 0.5*(-(2/tau)*log(2)+lgamma(1/tau)-lgamma(3/tau))
	 		logC[i] = 0.5 * (-(2 / tauV.getEntry(i)) * FastMath.log(2) 
	 								+ Gamma.logGamma(1 / tauV.getEntry(i)) 
	 								   - Gamma.logGamma(3 / tauV.getEntry(i)));

	 		c[i] = FastMath.exp(logC[i]);
	 	}
	 	
	 	if (whichDistParameter == DistributionSettings.TAU){
	 		dlogcDt = new double[size];
		 	for (int i = 0; i < size; i++) {
		 		//dlogc.dt <- (1/(2*tau^2))*(2*log(2)
		 		//-digamma(1/tau)+3*digamma(3/tau)) 
		 		dlogcDt[i] = (1 / (2 * tauV.getEntry(i) 
		 	    				        * tauV.getEntry(i))) * (2 
		 	    						* FastMath.log(2) - Gamma.digamma(1 
		 	    					/ tauV.getEntry(i)) + 3 * Gamma.digamma(
		 	    										3 / tauV.getEntry(i)));
		 	}
	 	}
	}

	
	
	/**  First derivative dldm = dl/dmu, where l - log-likelihood function.
	 * @param y - vector of values of response variable
	 * @return  a vector of first derivative dldm = dl/dmu
	 */	
	public final ArrayRealVector dldm(final ArrayRealVector y) {
		
		dldm = new double[size];
	 	for (int i = 0; i < size; i++) {
	 		
	 	   //dldm <- (tau/(2*mu*sigma*c^2))*(z+sigma*nu*z^2)
	 	   //*((abs(z/c))^(tau-2))-(nu/mu)
	 		dldm[i] = (tauV.getEntry(i) / (2 * muV.getEntry(i) 
	 				* sigmaV.getEntry(i) * c[i] * c[i]))
	 				  * (z[i] + sigmaV.getEntry(i) * nuV.getEntry(i)
	 						* z[i] * z[i]) * (FastMath.pow(FastMath.abs(z[i]
	 							/ c[i]), (tauV.getEntry(i) - 2)))
	 									- (nuV.getEntry(i)  / muV.getEntry(i));
	 	}
		c = null;
		logC = null;
		z = null;
	 	 return new ArrayRealVector(dldm, false);
	}
	
	/** First derivative dlds = dl/dsigma, 
	 * where l - log-likelihood function.
	 * @param y - vector of values of response variable
	 * @return  a vector of First derivative dlds = dl/dsigma
	 */	
	public final ArrayRealVector dlds(final ArrayRealVector y) {

		dlds = new double[size];
	 	for (int i = 0; i < size; i++) {

		 	 //h <- f.T(1/(sigma*abs(nu)))/F.T(1/(sigma*abs(nu)))
	 	     final double h = f1T(1 / (sigmaV.getEntry(i) 
	 	    		 * FastMath.abs(nuV.getEntry(i))), tauV.getEntry(i),
	 	    		 				false, i) / f2T(1 / (sigmaV.getEntry(i) 
	 	    				 				* FastMath.abs(nuV.getEntry(i))), 
	 	    				 							  tauV.getEntry(i), i);
	 	     
		 	//dldd <- (1/sigma)*((tau/2)*(abs(z/c))^tau-1)+h/(sigma^2*abs(nu))
	 	    dlds[i] = (1 / sigmaV.getEntry(i)) * ((tauV.getEntry(i) / 2)
	 	    		* FastMath.pow(FastMath.abs(z[i] / c[i]), tauV.getEntry(i))
	 	    		        - 1) + h / (sigmaV.getEntry(i) * sigmaV.getEntry(i)
	 	    								  * FastMath.abs(nuV.getEntry(i)));
	 		
	 	}
		c = null;
		logC = null;
		z = null;
	  	return new ArrayRealVector(dlds, false);
	}
	
	/**
	 * Supportive function to compute first derivatives.
	 * @param t -  value
	 * @param tau -  value of tau distribution parameter
	 * @param isLog - whether likelihood function is a log
	 * @param i - loop count
	 *  of likelihood function or not
	 * @return value of likelihood function
	 */
	private double f1T(final double t, 
					   final double tau, 
					   final boolean isLog,
					   final int i) {
		    
		    //loglik <- log(tau)-log.c-(0.5*(abs(t/c)^tau))
			//-(1+(1/tau))*log(2)-lgamma(1/tau)
		    final double out  = FastMath.log(tau) - logC[i] - (0.5 
		    		 		  * (FastMath.pow(FastMath.abs(t / c[i]), tau)))
		    		 		  			  - (1 + (1 / tau)) * FastMath.log(2)
		    										 - Gamma.logGamma(1 / tau);
		    
		   //if(log==FALSE) fT  <- exp(loglik) else fT <- loglik
		    if (!isLog) {
		    	
		    	return FastMath.exp(out);
		    } else {
		    	return out;
		    }
	}
	
	/**
	 * Supportive function to compute first derivatives.
	 * @param t -  value
	 * @param tau -  value of tau distribution parameter
	 * @return modified value of regularized gamma function 
	 * @param i - loop count
	 */
	private double f2T(final double t, final double tau, final int i) {
				    
		    //s <- 0.5*((abs(t/c))^tau)
		    final double s =  0.5 * ((FastMath.pow(FastMath.abs(t
		    											/ c[i]), tau)));
		    
		    //F.s <- pgamma(s,shape = 1/tau, scale = 1)
		    gammaDistr.setDistrParameters(1 / tau, 1.0);
		    final double fS = gammaDistr.cumulativeProbability(s);
		    
		    //cdf <- 0.5*(1+F.s*sign(t))
		    return 0.5 * (1 + fS * FastMath.signum(t));
	}
	
	/**
	 * Supportive function to compute first derivatives. 
	 * @param t -  value
	 * @param tau -  value of tau distribution parameter
	 * @param isLog - whether likelihood function is a
	 *  log of likelihood function or not
	 * @return value of likelihood function
	 */

	/** First derivative dldn = dl/dnu,
	 *  where l - log-likelihood function.
	 * @param y - vector of values of response variable
	 * @return  a vector of First derivative dldn = dl/dnu
	 */	
	public final ArrayRealVector dldn(final ArrayRealVector y) {

		dldn = new double[size];
	 	for (int i = 0; i < size; i++) {

		 	 //h <- f.T(1/(sigma*abs(nu)))/F.T(1/(sigma*abs(nu)))
	 	     final double h = f1T(1 / (sigmaV.getEntry(i) 
	 	    		 * FastMath.abs(nuV.getEntry(i))), tauV.getEntry(i),
	 	    		 				false, i) / f2T(1 / (sigmaV.getEntry(i) 
	 	    				 				* FastMath.abs(nuV.getEntry(i))), 
	 	    				 							  tauV.getEntry(i), i);
	 	     
		 	 //dldv <-  -(tau/(2*nu*c^2))*((abs(z/c))^(tau-2))
	 	     //*z*((nu*z+1/sigma) *log(y/mu)-z)
	 	     dldn[i] =  -(tauV.getEntry(i) / (2 * nuV.getEntry(i) 
	 	    	 * c[i] * c[i])) * (FastMath.pow((FastMath.abs( z[i]
	 	    				   / c[i])), (tauV.getEntry(i)-2))) * z[i] 
	 	    						 	* ((nuV.getEntry(i) * z[i] + 1 
	 	    								 		/ sigmaV.getEntry(i)) 
	 	    								 	* FastMath.log(y.getEntry(i) 
	 	    										/ muV.getEntry(i)) - z[i]);
		 	 
	 	     //dldv <- dldv+log(y/mu)+sign(nu)*h/(sigma*nu^2)
	 	    dldn[i] = dldn[i] + FastMath.log(y.getEntry(i) 
	 	    		/ muV.getEntry(i)) + FastMath.signum(nuV.getEntry(i))
	 	    					 * h / (sigmaV.getEntry(i) * nuV.getEntry(i) 
	 	    												* nuV.getEntry(i));
	 	}
		c = null;
		logC = null;
		z = null;
	  	return new ArrayRealVector(dldn, false);
	}

	/** First derivative dldtau = dl/dtau,
	 *  where l - log-likelihood function.
	 * @param y - vector of values of response variable
	 * @return  a vector of First derivative dldtau = dl/dtau
	 */
	public final ArrayRealVector dldt(final ArrayRealVector y) {
		
		dldt = new double[size];
	 	for (int i = 0; i < size; i++) {

	 	    //j <- (log(F.T(1/(sigma*abs(nu)),tau+0.001))
	 	    //-log(F.T(1/(sigma*abs(nu)),tau)))/0.001
	 	   final double j = (FastMath.log(f2T(1 / (sigmaV.getEntry(i) 
	 			      * FastMath.abs(nuV.getEntry(i))), tauV.getEntry(i)
	 			    + 0.001, i)) - FastMath.log(f2T(1 / (sigmaV.getEntry(i)
	 					   * FastMath.abs(nuV.getEntry(i))), tauV.getEntry(i),
	 					   										i))) / 0.001;
	 	    
	 	    //dldt <- (1/tau)-0.5*(log(abs(z/c)))*
	 	    //(abs(z/c))^tau+(1/tau^2)*(log(2)+digamma(1/tau))+((tau/2)*
	 	   //((abs(z/c))^tau)-1)*dlogc.dt-j
	 	  dldt[i] = (1 / tauV.getEntry(i)) - 0.5
	 			  * (FastMath.log(FastMath.abs(z[i] / c[i]))) 
	 			  * (FastMath.pow(FastMath.abs(z[i] / c[i]), tauV.getEntry(i)))
	 			  				+ (1 / (tauV.getEntry(i) * tauV.getEntry(i))) 
	 			  	  * (FastMath.log(2) + Gamma.digamma(1 / tauV.getEntry(i))) 
	 			   + ((tauV.getEntry(i) / 2) * (FastMath.pow(FastMath.abs(z[i] 
	 					  	/ c[i]), tauV.getEntry(i))) - 1) * dlogcDt[i] - j;
	 	}
		c = null;
		logC = null;
		z = null;
	  	return new ArrayRealVector(dldt, false);
	}

	/** Calculates a second derivative of the likelihood 
	 * function in respect to supplied distribution parameter.
	 * @param whichDistParameter - distribution parameter
	 * @param y - vector of values of likelihood function 
	 * @return vector of second derivative of the likelihood
	 */
	public final ArrayRealVector secondDerivative(final int whichDistParameter,
												  final ArrayRealVector y) {
				tempV = null;		
				switch (whichDistParameter) {
			      case DistributionSettings.MU:
			    	   tempV = d2ldm2(y);
			          break;
			      case DistributionSettings.SIGMA:
			    	 tempV = d2lds2(y);
			    	 break;
			      case DistributionSettings.NU:
			    	 tempV = d2ldn2(y);
			    	 break;
			      case DistributionSettings.TAU:
			    	 tempV = d2ldt2(y);
			    	 break;
			      default: 
					System.err.println("Requested second order "
							+ "derivative does not exist");
					break;
			    }
				return tempV;
	}


	/** Second derivative d2ldm2= (d^2l)/(dmu^2), 
	 * where l - log-likelihood function.
	 * @param y - vector of values of response variable
	 * @return  a vector of second derivative d2ldm2= (d^2l)/(dmu^2)
	*/
	private ArrayRealVector d2ldm2(final ArrayRealVector y) {	
		
		double[] out = new double[size];
	 	for (int i = 0; i < size; i++) {
	 		
	 		//d2ldm2 <- if (any(tau<1.05)) -dldm*dldm else d2ldm2
	 		if (tauV.getEntry(i) < 1.05) {
	 			out[i] = -dldm[i] * dldm[i];
	 		} else {
	 			
	 		   //d2ldm2 <- -((tau*tau)*gamma(2-(1/tau))*gamma(3/tau))
	 		   ///(mu^2*sigma^2*(gamma(1/tau))^2)
	 		   //d2ldm2 <- d2ldm2-(tau*nu^2)/mu^2
	 		out[i] = -((tauV.getEntry(i) * tauV.getEntry(i)) 
	 		     * FastMath.exp(Gamma.logGamma(2 - (1 / tauV.getEntry(i))))
	 			      * FastMath.exp(Gamma.logGamma(3 / tauV.getEntry(i)))) 
	 				/ (muV.getEntry(i) * muV.getEntry(i) * sigmaV.getEntry(i)
	 					* sigmaV.getEntry(i) * (FastMath.exp(Gamma.logGamma(1 
	 				    / tauV.getEntry(i)))) * (FastMath.exp(Gamma.logGamma(1 
	 													/ tauV.getEntry(i)))));
	 		out[i] = out[i] - (tauV.getEntry(i) * nuV.getEntry(i) 
	 				* nuV.getEntry(i)) / (muV.getEntry(i) * muV.getEntry(i));
	 		}
	 	 }
	 	 muV     = null;
	 	 sigmaV  = null;
	 	 nuV     = null;
	 	 tauV    = null;
	 	 return new ArrayRealVector(out, false);
	}	

	/** Second derivative d2lds2= (d^2l)/(dsigma^2),
	 *  where l - log-likelihood function.
 * @param y - vector of values of response variable
	 * @return  a vector of second derivative d2lds2= (d^2l)/(dsigma^2)
	 */
	private ArrayRealVector d2lds2(final ArrayRealVector y) {
		
		double[] out = new double[size];
	 	for (int i = 0; i < size; i++) {

	 		//d2ldd2 = function(sigma,tau) -tau/sigma^2
	 		out[i] = -tauV.getEntry(i) / (sigmaV.getEntry(i) 
	 											* sigmaV.getEntry(i));
	 		
	 		//out[i] = -((tauV.getEntry(i)) / sigmaV.getEntry(i)) / sigmaV.getEntry(i);
	 	 }
	 	 muV     = null;
	 	 sigmaV  = null;
	 	 nuV     = null;
	 	 tauV    = null;
	  	return new ArrayRealVector(out, false);
	}	

	/** Second derivative d2ldn2= (d^2l)/(dnu^2),
	 *  where l - log-likelihood function.
	 * @param y - vector of values of response variable
	 * @return  a vector of second derivative d2ldn2= (d^2l)/(dnu^2)
	 */
	private ArrayRealVector d2ldn2(final ArrayRealVector y) {	
		
		double[] out = new double[size];
	 	for (int i = 0; i < size; i++) {

	 	    //d2ldv2 = function(sigma,tau) { -sigma^2*(3*tau+1)/4}
	 		out[i] = -sigmaV.getEntry(i) * sigmaV.getEntry(i) 
	 								* (3 * tauV.getEntry(i) + 1) / 4;
	 	 }
	 	 muV     = null;
	 	 sigmaV  = null;
	 	 nuV     = null;
	 	 tauV    = null;
	  	return new ArrayRealVector(out, false);
	}	

	/** Second derivative d2ldt2= (d^2l)/(dtau^2), 
	 * where l - log-likelihood function.
	 * @param y - vector of values of response variable
	 * @return  a vector of second derivative d2ldt2= (d^2l)/(dtau^2)
	 */ 
	private ArrayRealVector d2ldt2(final ArrayRealVector y) {	

		double[] out = new double[size];
	 	for (int i = 0; i < size; i++) {
	        //p <- (tau+1)/tau
	        final double p = (tauV.getEntry(i) + 1) / tauV.getEntry(i);
	 		
	 		//part1 <- p*trigamma(p)+2*(digamma(p))^2
	        final double part1 = p * Gamma.trigamma(p) 
	        		+ 2 * (Gamma.digamma(p)) * (Gamma.digamma(p));
	 		
	 		//part2 <- digamma(p)*(log(2)+3-3*digamma(3/tau)-tau)
	        final double part2 = Gamma.digamma(p) * (FastMath.log(2) 
	        				+ 3 - 3 * Gamma.digamma(3 / tauV.getEntry(i)) 
	        											   - tauV.getEntry(i));
	 		
	 		//part3 <- -3*(digamma(3/tau))*(1+log(2))    
	        final double part3 = -3 * (Gamma.digamma(3 / 
	        						tauV.getEntry(i))) * (1 + FastMath.log(2));
	 		
	 		//part4 <- -(tau+log(2))*log(2)
	        final double part4 = -(tauV.getEntry(i) + FastMath.log(2)) 
	        												*FastMath.log(2);
	 		
	 		//part5 <- -tau+(tau^4)*(dlogc.dt)^2
	        final double part5 = -tauV.getEntry(i) 
	        					+ (FastMath.pow(tauV.getEntry(i), 4)) 
	        									* (dlogcDt[i]) * (dlogcDt[i]);
	        
	 		
	 		//d2ldt2 <- part1+part2+part3+part4+part5
	        out[i] = part1+ part2 + part3 + part4 +part5;
	 		
	 		//d2ldt2 <- -d2ldt2/tau^3    
	        out[i] = -out[i]/(tauV.getEntry(i)*tauV.getEntry(i)*tauV.getEntry(i));
	 		
	 		//d2ldt2 <- ifelse(d2ldt2 < -1e-15, d2ldt2,-1e-15)
	        if(out[i] > -1e-15)
	        {
	        	out[i] = -1e-15;
	        }
	 		
	 	 }
	 	 muV     = null;
	 	 sigmaV  = null;
	 	 nuV     = null;
	 	 tauV    = null;
	  	return new ArrayRealVector(out, false);
	}
	
	/** Calculates a second cross derivative of the likelihood 
	 * function in respect to supplied distribution parameters.
	 * @param whichDistParameter1 - first distribution parameter
	 * @param whichDistParameter2 - second distribution parameter
	 * @param y - vector of values of likelihood function 
	 * @return  vector of second cross derivative of the likelihood
	 */
	public final ArrayRealVector secondCrossDerivative(
												 final int whichDistParameter1,
												 final int whichDistParameter2, 
												 final ArrayRealVector y) {	
		tempV = null;		
		if (whichDistParameter1 == DistributionSettings.MU) {
			switch (whichDistParameter2) {
			case DistributionSettings.SIGMA:
				tempV = d2ldmds(y);                      
				break;
			case DistributionSettings.NU:
				tempV = d2ldmdn(y);
				break;
			case DistributionSettings.TAU:
				tempV = d2ldmdt(y);
				break;
	          default: 
		  		System.err.println("Second derivative does not exist");
		  		return null;
			}
		}
		if (whichDistParameter1 == DistributionSettings.SIGMA) {
			switch (whichDistParameter2) {
			case DistributionSettings.NU:
				tempV = d2ldsdn(y);
				break;
			case DistributionSettings.TAU:
				tempV = d2ldsdt(y);
				break;
	          default: 
		  		System.err.println("Second derivative does not exist");
		  		return null;
			}
		}
		if (whichDistParameter1 == DistributionSettings.NU) {
			switch (whichDistParameter2) {
			case DistributionSettings.TAU:
				tempV = d2ldndt(y);
				break;
	          default: 
	  			System.err.println("Second derivative does not exist");
	  			return null;
			}
		}
		return tempV;
		}


	/** Second cross derivative of likelihood function
	 *  in respect to mu and sigma (d2ldmdd = d2l/dmu*dsigma).
	 * @param y - vector of values of response variable
	 * @return  a vector of Second cross derivative
	 */
	private ArrayRealVector d2ldmds(final ArrayRealVector y) {
	
		ArrayRealVector mu     
		= distributionParameters.get(DistributionSettings.MU);
		ArrayRealVector sigma  
		= distributionParameters.get(DistributionSettings.SIGMA);
		ArrayRealVector nu     
		= distributionParameters.get(DistributionSettings.NU);
		ArrayRealVector tau    
		= distributionParameters.get(DistributionSettings.TAU);
	 
	 	size = y.getDimension();
		double[] out = new double[size];
	 	for (int i = 0; i < size; i++) {
	 		//d2ldmdd = function(mu,sigma,nu,tau) -(nu*tau)/(mu*sigma)
	 		out[i] = -(nu.getEntry(i) * tau.getEntry(i))
	 							/ (mu.getEntry(i) * sigma.getEntry(i));
	 	 }
	  	return new ArrayRealVector(out, false);
	}
	
	/** Second cross derivative of likelihood function
	 *  in respect to mu and nu (d2ldmdd = d2l/dmu*dnu).
	 * @param y - vector of values of response variable
	 * @return  a vector of Second cross derivative
	 */
	private ArrayRealVector d2ldmdn(final ArrayRealVector y) {	

		ArrayRealVector mu     
		= distributionParameters.get(DistributionSettings.MU);
		ArrayRealVector sigma  
		= distributionParameters.get(DistributionSettings.SIGMA);
		ArrayRealVector nu     
		= distributionParameters.get(DistributionSettings.NU);
		ArrayRealVector tau    
		= distributionParameters.get(DistributionSettings.TAU);
	 
	 	size = y.getDimension();
	 	double[] out = new double[size];
	 	for (int i = 0; i < size; i++) {
	 		//d2ldmdv = function(mu,sigma,nu,tau) 
	 		//(2*(tau-1)-(tau+1)*(sigma^2)*(nu^2))/(4*mu),
	 		out[i] = (2 * (tau.getEntry(i) - 1) - (tau.getEntry(i)
	 				    + 1) * (sigma.getEntry(i) * sigma.getEntry(i))
	 								* (nu.getEntry(i) * nu.getEntry(i)))
	 												/ (4 * mu.getEntry(i));
	 	 }
	  	return new ArrayRealVector(out, false);
	}
	
	/** Second cross derivative of likelihood function 
	 * in respect to mu and tau (d2ldmdd = d2l/dmu*dtau).
	 * @param y - vector of values of response variable
	 * @return  a vector of Second cross derivative
	 */
	private ArrayRealVector d2ldmdt(final ArrayRealVector y) {
		
		ArrayRealVector mu     
		= distributionParameters.get(DistributionSettings.MU);
		ArrayRealVector sigma  
		= distributionParameters.get(DistributionSettings.SIGMA);
		ArrayRealVector nu     
		= distributionParameters.get(DistributionSettings.NU);
		ArrayRealVector tau    
		= distributionParameters.get(DistributionSettings.TAU);
	 
	 	size = y.getDimension();
	 	double[] out = new double[size];
	 	for (int i = 0; i < size; i++) {
	 		//d2ldmdt = (nu/(mu*tau))*(1+tau+(3/2)
	 		//*(digamma(1/tau)-digamma(3/tau)))
	 		out[i] = (nu.getEntry(i) / (mu.getEntry(i) * tau.getEntry(i))) 
	 				     * (1 + tau.getEntry(i) + (3 / 2) * (Gamma.digamma(1
	 									  / tau.getEntry(i)) - Gamma.digamma(3
	 											  		/ tau.getEntry(i))));
	 		
	 	 }
	  	return new ArrayRealVector(out, false);
	}
	

	/** Second cross derivative of likelihood function 
	 * in respect to sigma and nu (d2ldmdd = d2l/dsigma*dnu).
	 * @param y - vector of values of response variable
	 * @return  a vector of Second cross derivative
	 */
	private ArrayRealVector d2ldsdn(final ArrayRealVector y) {	

		ArrayRealVector mu     
		= distributionParameters.get(DistributionSettings.MU);
		ArrayRealVector sigma  
		= distributionParameters.get(DistributionSettings.SIGMA);
		ArrayRealVector nu     
		= distributionParameters.get(DistributionSettings.NU);
		ArrayRealVector tau    
		= distributionParameters.get(DistributionSettings.TAU);
	 
	 	size = y.getDimension();
		double[] out = new double[size];
	 	for (int i = 0; i < size; i++) {
	 		//2ldddv = function(sigma,nu,tau) -(sigma*nu*tau)/2
	 		out[i] = -(sigma.getEntry(i) * nu.getEntry(i) 
	 									* tau.getEntry(i)) / 2;
	
	 	 }
	  	return new ArrayRealVector(out, false);
	}

	/** Second cross derivative of likelihood function
	 *  in respect to sigma and tau (d2ldmdd = d2l/dsigma*dtau).
	 * @param y - vector of values of response variable
	 * @return  a vector of Second cross derivative
	 */
	private ArrayRealVector d2ldsdt(final ArrayRealVector y) {	

		ArrayRealVector mu     
		= distributionParameters.get(DistributionSettings.MU);
		ArrayRealVector sigma  
		= distributionParameters.get(DistributionSettings.SIGMA);
		ArrayRealVector nu     
		= distributionParameters.get(DistributionSettings.NU);
		ArrayRealVector tau    
		= distributionParameters.get(DistributionSettings.TAU);
	 
	 	size = y.getDimension();
		double[] out = new double[size];
	 	for (int i = 0; i < size; i++) {
	 		//d2ldddt = (1/(sigma*tau))*(1+tau+(3/2)
	 		//*(digamma(1/tau)-digamma(3/tau)))
	 		out[i] = (1 / (sigma.getEntry(i) * tau.getEntry(i)))
	 				* (1 + tau.getEntry(i) + (3 / 2) * (Gamma.digamma(1
	 						            / tau.getEntry(i)) - Gamma.digamma(3
	 													/ tau.getEntry(i))));
	 		
	 		
	 	 }
	  	return new ArrayRealVector(out, false);
	}
	
	/** Second cross derivative of likelihood function 
	 * in respect to nu and tau (d2ldmdd = d2l/dnu*dtau).
	 * @param y - vector of values of response variable
	 * @return  a vector of Second cross derivative
	 */
	private ArrayRealVector d2ldndt(final ArrayRealVector y) {	

		ArrayRealVector mu     
		= distributionParameters.get(DistributionSettings.MU);
		ArrayRealVector sigma  
		= distributionParameters.get(DistributionSettings.SIGMA);
		ArrayRealVector nu     
		= distributionParameters.get(DistributionSettings.NU);
		ArrayRealVector tau    
		= distributionParameters.get(DistributionSettings.TAU);
	 
	 	size = y.getDimension();
		double[] out = new double[size];
	 	for (int i = 0; i < size; i++) {
	 		//d2ldvdt  <- (((sigma^2)*nu)/(2*tau))
	 		//*(1+(tau/3)+0.5*(digamma(1/tau)-digamma(3/tau)))
	 		out[i] = (((sigma.getEntry(i) * sigma.getEntry(i)) 
	 				* nu.getEntry(i)) / (2 * tau.getEntry(i)))
	 				* (1 + (tau.getEntry(i) / 3) + 0.5
	 						* (Gamma.digamma(1 / tau.getEntry(i))
	 								- Gamma.digamma(3 / tau.getEntry(i))));
	 	 }
	  	return new ArrayRealVector(out, false);
	}
	 		
	 		
	/** Computes the global Deviance Increament.
	 * @param y - vector of response variable values
	 * @return vector of global Deviance Increament values 
	 */
	public final ArrayRealVector globalDevianceIncreament(
													final ArrayRealVector y) {
      // G.dev.incr  = function(y,mu,sigma,nu,tau,...)
	  //-2*dBCPE(y,mu,sigma,nu,tau,log=TRUE),
	  size = y.getDimension();
	  double[] out = new double[size];

	  double[] muArr = distributionParameters.get(
									DistributionSettings.MU).getDataRef();
	  double[] sigmaArr = distributionParameters.get(
									DistributionSettings.SIGMA).getDataRef();
	  double[] nuArr = distributionParameters.get(
									DistributionSettings.NU).getDataRef();
	  double[] tauArr = distributionParameters.get(
									DistributionSettings.TAU).getDataRef();
	
	  for (int i = 0; i < size; i++) {
		
		out[i] = (-2) * dBCPE(y.getEntry(i), muArr[i], sigmaArr[i], 
							nuArr[i], tauArr[i], Controls.LOG_LIKELIHOOD);
	  }
	  return new ArrayRealVector(out, false);
	}

	/** Computes the probability density function (PDF) of this 
	 * distribution evaluated at the specified point x.
	 * @param x - value of response variable
	 * @param mu - value of mu distribution parameter
	 * @param sigma - value of sigma distribution parameter
	 * @param nu - value of nu distribution parameter
	 * @param tau - value of tau distribution parameter
	 * @param isLog  - logical, whether to take log of the function or not
	 * @return value of probability density function
	 */
	public final double dBCPE(final double x, 
			 			      final double mu, 
			 			      final double sigma, 
			 			      final double nu, 
			 			      final double tau, 
			 			      final boolean isLog) {
		
		// if (any(mu < 0))  stop(paste("mu must be positive", "\n", "")) 
		if (mu < 0) {
			System.err.println("mu must be positive");
			return -1.0;
		}
		
		//if (any(sigma < 0))  stop(paste("sigma must be positive", "\n", ""))
		if (sigma < 0) {
			System.err.println("sigma must be positive");
			return -1.0;
		}
		
		// if (any(tau < 0))  stop(paste("tau must be positive", "\n", ""))  
		if (tau < 0) {
			System.err.println("tau must be positive");
			return -1.0;
		}

		//if (any(x < 0))  stop(paste("x must be positive", "\n", "")) 
		if (x < 0) {
			System.err.println("x must be positive");
			return -1.0;
		}

		//if(length(nu)>1) z <- ifelse(nu != 0,
		//(((x/mu)^nu-1)/(nu*sigma)),log(x/mu)/sigma)
		double out = 0;
	
		//z <- ifelse(nu != 0,(((y/mu)^nu-1)/(nu*sigma)),log(y/mu)/sigma)
	    if (nu != 0) {
	    	
	    	out = (FastMath.pow(x / mu, nu) - 1) / (nu * sigma);
	    } else {
	    	
	    	out = FastMath.log(x / mu) / sigma;
	    }
	    
	    //logfZ <- f.T(z,log=TRUE)-log(F.T(1/(sigma*abs(nu))))
	    final double logfZ = f1T(out, tau, true) - FastMath.log(f2T(1 
	    						/ (sigma * FastMath.abs(nu)), tau));
	    
	    //logder <- (nu-1)*log(x)-nu*log(mu)-log(sigma)
	    final double logder = (nu - 1) * FastMath.log(x) - nu 
	    						* FastMath.log(mu) - FastMath.log(sigma);
	    
	    //loglik <- logder+logfZ
	    out = logfZ + logder;
	    
	    //if(log==FALSE) ft  <- exp(loglik) else ft <- loglik 
	    if (!isLog) {
	    	out = FastMath.exp(out);
	    }
	    return out;
	}

	/**
	 * dBCPE(x) launches dBCPE(x, mu, sigma, nu, isLog) 
	 * with deafult mu=5, sigma=0.1, nu=1, tau=2, log=FALSE.
	 * @param x - vector of response variable values
	 * @return vector of probability density function values
	 */
	//dBCPE <- function(x, mu=5, sigma=0.1, nu=1, tau=2, log=FALSE)
	public final double dBCPE(final double x) {
		return dBCPE(x, 5.0, 0.1, 1.0, 2.0, false);
	}	
	
	/**
	 * Supportive function to compute first derivatives.
	 * @param t -  value
	 * @param tau -  value of tau distribution parameter
	 * @param isLog - whether likelihood function is a log
	 *  of likelihood function or not
	 * @return value of likelihood function
	 */
	private double f1T(final double t, 
					   final double tau, 
					   final boolean isLog) {
		    
		//log.c <- 0.5*(-(2/tau)*log(2)+lgamma(1/tau)-lgamma(3/tau))
 		final double logC = 0.5 * (-(2 / tau) * FastMath.log(2) 
 								+ Gamma.logGamma(1 / tau) 
 								   - Gamma.logGamma(3 / tau));
	
 		//c <- exp(log.c)
 		final double  c = FastMath.exp(logC);
	
	    //loglik <- log(tau)-log.c-(0.5*(abs(t/c)^tau))
		//-(1+(1/tau))*log(2)-lgamma(1/tau)
	    final double out  = FastMath.log(tau) - logC - (0.5 
	    		 		  * (FastMath.pow(FastMath.abs(t / c), tau)))
	    		 		  			  - (1 + (1 / tau)) * FastMath.log(2)
	    										 - Gamma.logGamma(1 / tau);
	    
	   //if(log==FALSE) fT  <- exp(loglik) else fT <- loglik
	    if (!isLog) {
	    	
	    	return FastMath.exp(out);
	    } else {
	    	
	    	return out;
	    }
	}
	
	/**
	 * Supportive function to compute first derivatives.
	 * @param t -  value
	 * @param tau -  value of tau distribution parameter
	 * @return modified value of regularized gamma function 
	 */
	private double f2T(final double t, final double tau) {
		
		//log.c <- 0.5*(-(2/tau)*log(2)+lgamma(1/tau)-lgamma(3/tau))	
 		//c <- exp(log.c)
 		final double  c = FastMath.exp(0.5 * (-(2 / tau) * FastMath.log(2)
											+ Gamma.logGamma(1 / tau) 
											- Gamma.logGamma(3 / tau)));
    
	    //s <- 0.5*((abs(t/c))^tau)
	    final double s =  0.5 * ((FastMath.pow(FastMath.abs(t
	    											/ c), tau)));
	    
	    //F.s <- pgamma(s,shape = 1/tau, scale = 1)
	    gammaDistr.setDistrParameters(1 / tau, 1.0);
	    final double fS = gammaDistr.cumulativeProbability(s);
	    
	    //cdf <- 0.5*(1+F.s*sign(t))
	    return 0.5 * (1 + fS * FastMath.signum(t));
	}


	/** Computes the cumulative distribution 
	 * function P(X <= q) for a random variable X .
	 * whose values are distributed according to this distribution
	 * @param q - value of quantile
	 * @param mu - value of mu distribution parameter
	 * @param sigma - value of sigma distribution parameter
	 * @param nu - value of nu distribution parameter 
	 * @param tau - value of tau distribution parameter 
	 * @param lowerTail - logical, if TRUE (default), probabilities
	 *  are P[X <= x] otherwise, P[X > x].
	 * @param isLog - logical, if TRUE, probabilities p are given as log(p)
	 * @return value of cumulative probability function values P(X <= q)
	 */
	public final double pBCPE(final double q, 
 	   		                  final double mu, 
 	   		                  final double sigma, 
 	   		                  final double nu,
 	   		                  final double tau, 
 	   		                  final boolean lowerTail, 
 	   		                  final boolean isLog) {
		// if (any(mu < 0))  stop(paste("mu must be positive", "\n", "")) 
		if (mu < 0) {
			System.err.println("mu must be positive");
			return -1.0;
		}
		
		//if (any(sigma < 0))  stop(paste("sigma must be positive", "\n", "")) 
		if (sigma < 0) {
			System.err.println("sigma must be positive");
			return -1.0;
		}
		
		// if (any(tau < 0))  stop(paste("tau must be positive", "\n", ""))  
		if (tau < 0) {
			System.err.println("tau must be positive");
			return -1.0;
		}

		//if (any(q < 0))  stop(paste("x must be positive", "\n", "")) 
		if (q < 0) {
			System.err.println("q must be positive");
			return -1.0;
		}

		double out = 0;

		// z <- ifelse(nu != 0,(((q/mu)^nu-1)/(nu*sigma)),log(q/mu)/sigma)
	    if (nu != 0) {
	    	out = (FastMath.pow(q / mu, nu) - 1)/(nu * sigma);
	    } else {
	    	out = FastMath.log(q / mu) / sigma;
	    }
	    
        //FYy1 <- F.T(z,tau)
	    final double fyY1  = f2T(out, tau);
	    
	    //if(length(nu)>1)  FYy2 <- ifelse(nu >
	    //0, F.T( -1/(sigma*abs(nu)),tau),0)
	    double fyY2 = 0;
	    if (nu > 0) {
	    	
	    	fyY2 = f2T( -1 / (sigma * FastMath.abs(nu)), tau);
	    }
	  
	    //FYy3 <- F.T(1/(sigma*abs(nu)),tau)
	    final double fyY3 = f2T(1 / (sigma * FastMath.abs(nu)), tau);
	    
	    //FYy  <- (FYy1-FYy2)/FYy3
	    out = (fyY1 - fyY2) / fyY3;
	    
	    //if(lower.tail==TRUE) FYy  <- FYy else  FYy <- 1-FYy 
	    //if(log.p==FALSE) FYy  <- FYy else  FYy<- log(FYy) 
	    if (!lowerTail) {
	    	if (isLog) {
	    		
	    		out = FastMath.log(1 - out);
	    	} else {
	    		out = 1 - out;
	    	}
	    } else if (isLog) {
	    	out = FastMath.log(out);
	    }
	    return out;
	}
	
	/**
	 * pBCPE(q) launches pBCPE(q, mu, sigma, nu,  tau, lowerTail, isLog)
	 *  with deafult mu=5, sigma=0.1, nu=1, tau=2.
	 * lowerTail = true, isLog = false.
	 * @param q - quantile
	 * @return value of cumulative probability function P(X <= q)
	 */
	//pBCPE <- function(q, mu=5, sigma=0.1, nu=1, 
	//tau=2, lower.tail = TRUE, log.p = FALSE)
	public final double pBCPE(final double q) {
		return pBCPE(q, 5.0, 0.1, 1.0, 2.0, true, false);
	}
	
	/** Computes the quantile (inverse cumulative probability)
	 *  function  of this distribution.
	* @param p - vector of cumulative probability values
	* @param mu -  vector of mu distribution parameters values
	* @param sigma -  vector of sigma distribution parameters values
	* @param nu -  vector of nu distribution parameters values
	* @param tau -  vector of tau distribution parameters values
	* @param lowerTail - logical; if TRUE (default), probabilities 
	* are P[X <= x] otherwise, P[X > x]
	* @param isLog - logical; if TRUE, probabilities p are given as log(p).
	* @return vector of quantile function values
	*/
	public final double qBCPE(final double p, 
		 	                  final double  mu, 
		 	                  final double  sigma, 
		 	                  final double nu, 
		 	                  final double tau, 
		 	                  final boolean lowerTail, 
		 	                  final boolean isLog) {
	
		// if (any(mu < 0))  stop(paste("mu must be positive", "\n", "")) 
		if (mu < 0) {
			System.err.println("mu must be positive");
			return -1.0;
		}
		
		//if (any(sigma < 0))  stop(paste("sigma must be positive", "\n", "")) 
		if (sigma < 0) {
			System.err.println("sigma must be positive");
			return -1.0;
		}
		
		// if (any(tau < 0))  stop(paste("tau must be positive", "\n", ""))  
		if (tau < 0) {
			System.err.println("tau must be positive");
			return -1.0;
		}
	
		double out = p;
		
		// if (log.p==TRUE) p <- exp(p) else p <- p
		if (isLog) {
			out = FastMath.exp(out);
		}
		
		//if (any(p <= 0)|any(p >= 1))  stop(paste("p must be between 0 and 1)
		if (out <= 0 || out >= 1) {
			System.err.println("p must be between 0 and 1");
		}
		if (!lowerTail) {
			out =  1 - out;
		}
		
		//za <- ifelse(nu<0,  q.T(p*F.T(1/(sigma*abs(nu)),tau),
		//tau),q.T((1-(1-p)*F.T(1/(sigma*abs(nu)),tau)),tau))
		double za = 0;
		if (nu < 0) {
			
			za = qT(out * f2T(1 / (sigma * FastMath.abs(nu)), tau), tau);
		} else if(nu != 0) {
			//q.T((1-(1-p)*F.T(1/(sigma*abs(nu)),tau)),tau)
			za = qT((1 - (1 - out) * f2T(1 
					/ (sigma * FastMath.abs(nu)), tau)), tau);
		} else {
			//za <- ifelse(nu==0, q.T(p,tau), za)
			za = qT(out, tau);
		}
		
		//ya <- ifelse(nu != 0,mu*(nu*sigma*za+1)^(1/nu),mu*exp(sigma*za))
		if (nu != 0) {
			
			out = mu * FastMath.pow((nu * sigma * za + 1), (1 / nu));
		} else {
			
			out = mu * FastMath.exp(sigma * za);
		}
	 	return out;
	}
	
	/**
	 * Supportive function for qBCPE.
	 * @param p - cumulative probability value
	 * @param tau - tau distribution parameter
	 * @return value
	 */
	private double qT(final double p, final double tau) {
		
		//log.c <- 0.5*(-(2/tau)*log(2)+lgamma(1/tau)-lgamma(3/tau))	
 		//c <- exp(log.c)
 		final double  c = FastMath.exp(0.5 * (-(2 / tau) * FastMath.log(2)
											+ Gamma.logGamma(1 / tau) 
											- Gamma.logGamma(3 / tau)));
		    
		//s <- qgamma((2*p-1)*sign(p-0.5),shape=(1/tau),scale=1)
	    gammaDistr.setDistrParameters(1 / tau, 1.0);
	    final double s = gammaDistr.cumulativeProbability((2
	    		* p - 1) * FastMath.signum(p - 0.5));

		//z <- sign(p-0.5)*((2*s)^(1/tau))*c
		return FastMath.signum(p - 0.5) 
				* (FastMath.pow((2 * s), (1 / tau))) * c;
	}
	
	/**
	 * qBCPE(p) launches qBCPE(p, mu, sigma, nu,  tau, lowerTail, 
	 * isLog) with deafult mu=5, sigma=0.1, nu=1, tau=2,.
	 * lowerTail = true, isLog = false.
	 * @param p - cumulative probability value
	 * @return quantile function value
	 */
	//qBCPE <-  function(p, mu=5, sigma=0.1, nu=1,
	//tau=2, lower.tail = TRUE, log.p = FALSE)
	public double qBCPE(final double p) {
		return qBCPE(p, 5.0, 0.1, 1.0, 2.0, true, false);
	}
	

	/** Generates a random sample from this distribution.
	 * @param mu -  vector of mu distribution parameters values
	 * @param sigma -  vector of sigma distribution parameters values
	 * @param nu -  vector of nu distribution parameters values
	 * @param tau -  vector of tau distribution parameters values
	 * @param uDist -  object of UniformRealDistribution class;
	 * @return random sample vector
	 */	
	public final double rBCPE(final double mu, 
	       	                  final double sigma, 
	       	                  final double nu, 
	       	                  final double tau,
	       	                  final UniformRealDistribution uDist) {
	
			// if (any(mu < 0))  stop(paste("mu must be positive", "\n", "")) 
			if (mu < 0) {
				System.err.println("mu must be positive");
				return -1.0;
			}
			
			//if (any(sigma < 0))  stop(paste("sigma must be positive")
			if (sigma < 0) {
				System.err.println("sigma must be positive");
				return  -1.0;
			}
			
			// if (any(tau < 0))  stop(paste("tau must be positive"))  
			if (tau < 0) {
				System.err.println("tau must be positive");
				return  -1.0;
			}
			
			//n <- ceiling(n)
			//temp = FastMath.ceil(n.getEntry(i));			    
			//r <- qBCPE(p,mu=mu,sigma=sigma,nu=nu,tau=tau)
			return qBCPE(uDist.sample(), mu, sigma, nu, tau, true, false);
		}

	/**
	* rBCPE(n) launches rBCPE(n, mu, sigma, nu,  tau) 
	* with deafult mu=5, sigma=0.1, nu=1, tau=2.
	 * @param uDist -  object of UniformRealDistribution class;
	 * @return random sample vector
	*/
	//rBCPE <- function(n, mu=5, sigma=0.1, nu=1, tau=2)
	public final double rBCPE(final UniformRealDistribution uDist) {
		return rBCPE(5.0, 0.1, 1.0, 2.0, uDist);
	}
	
	/**
	* Checks whether the mu distribution parameter is valid.
	* @param y - vector of response variavbles
	* @return - boolean
	*/
	public final boolean isYvalid(final ArrayRealVector y) {
		return y.getMinValue() > 0;	
	}

	/** Checks whether entries of ArrayRealVectors 
	 * of distribution parameters are valid.
	* @param whichDistParameter - distribution parameter
	  @return Hashtable of booleans
	 */
	public final boolean areDistributionParametersValid(
											final int whichDistParameter) {
		boolean tempB = false;
		switch (whichDistParameter) {
      case DistributionSettings.MU:
      	tempB = isMuValid(
      				distributionParameters.get(DistributionSettings.MU));
         break;
      case DistributionSettings.SIGMA:
      	tempB = isSigmaValid(
      				distributionParameters.get(DistributionSettings.SIGMA));
        break;
      case DistributionSettings.NU:
      	tempB = isNuValid(
      				distributionParameters.get(DistributionSettings.NU));
        break;
      case DistributionSettings.TAU:
      	tempB = isTauValid(
      				distributionParameters.get(DistributionSettings.TAU));
        break;
		default: System.out.println("The specific distribution parameter"
				+ " does not exist for this distribution");
		}
		return tempB;
	}
	
	/**
	* Checks whether the mu distribution parameter is valid.
	* @param mu - vector of mu (mean) values
	* @return - boolean
	*/
	private boolean isMuValid(final ArrayRealVector mu) {
		return mu.getMinValue() > 0;
	}
	
	/**
	 * Checks whether the sigma distribution parameter is valid.
	 * @param sigma - vector of sigma (standard deviation) values
	 * @return - - boolean
	 */
	private boolean isSigmaValid(final ArrayRealVector sigma) {
		return sigma.getMinValue() > 0;	
	}	

	/**
	 * Checks whether the nu distribution parameter is valid.
	 * @param nu - vector of nu values
	 * @return - - boolean
	 */
	private boolean isNuValid(final ArrayRealVector nu) {
		//nu.valid = function(nu) TRUE 
		return true;	
	}

	/**
	 * Checks whether the tau distribution parameter is valid.
	 * @param tau - vector of nu values
	 * @return - - boolean
	 */
	private boolean isTauValid(final ArrayRealVector tau) {
		return tau.getMinValue()> 0;
	}
	
	/**
	 * Get number of distribution parameters.
	 * @return number of distribution parameters
	 */
	public final int getNumberOfDistribtionParameters() {
		return numDistPar;
	}

	/**
	 * Get type of distributuion (Continuous, Discrete or Mixed).
	 * @return type of distributuion
	 */
	public final int getTypeOfDistribution() {
		return DistributionSettings.CONTINUOUS;
	}

	/**
	 * Get distribution name.
	 * @return distribution name.
	 */
	public final int getFamilyOfDistribution() {
		return DistributionSettings.BCPE;
	}

	/**
	 * Set distribution parsameters.
	 * @param whichDistParameter - the fitting distribution parameter
	 * @param fvDistributionParameter - vector of values of 
	 * fitting distribution parameter
	 */
	public final void setDistributionParameter(
							final int whichDistParameter, 
							final ArrayRealVector fvDistributionParameter) {
		this.distributionParameters.put(whichDistParameter,
													fvDistributionParameter);
	}

	/**
	 * Get distribution parsameters.
	 * @param whichDistParameter - distribution parameter
	 * @return - vector of distribution parameter values
	 */
	public final ArrayRealVector getDistributionParameter(
												final int whichDistParameter) {
		return this.distributionParameters.get(whichDistParameter);
	}

	/** Get the link function type of the current distribution parameter.
	 * @param whichDistParameter - distribution parameter
	 * @return link function type
	 */
	public final int getDistributionParameterLink(
												final int whichDistParameter) {
		return distributionParameterLink.get(whichDistParameter);
	}
}
