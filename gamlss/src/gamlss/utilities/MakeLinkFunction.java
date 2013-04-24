/*
  Copyright 2012 by Dr. Vlasios Voudouris and ABM Analytics Ltd
  Licensed under the Academic Free License version 3.0
  See the file "LICENSE" for more information
*/
package gamlss.utilities;

import gamlss.distributions.DistributionSettings;

import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.util.FastMath;
/**
 * 01/08/2012
 * @author Dr. Vlasios Voudouris, Daniil Kiose, Prof. Mikis Stasinopoulos and Dr Robert Rigby
 *
 */

public  class MakeLinkFunction 
{			
	public MakeLinkFunction(){
	}

//-------------------------------------------------------------------------------------------------		
	/**
	 * Checks whether supplied distribution parameter valid or not
	 * @param whichLink - type of link function
	 * @param eta - eta - vector of linear predictor values
	 * @return boolean
	 */
	public static boolean validate(int whichLink, ArrayRealVector eta)
	{		
		boolean out = false;
		switch (whichLink) 
		{
          case DistributionSettings.IDENTITY:
        	 out = identityValideta(eta);
             break;
          case DistributionSettings.LOG:
       	     out = logValideta(eta);
             break;
          case DistributionSettings.LOGSHIFTTO2:
        	     out = logShiftTo2Valideta(eta);
              break;
          default: 
			System.err.println("error: link function is not support");
			break;
         }
		return out;
	}

//-------------------------------------------------------------------------------------------------		
	/** Checks whether supplied link function is allowed for the current distribution
	 * @param whichDistribution - the distribution
	 * @param whichLink - type of link function
	 * @return -1 means that the specific distribution has not been implemented yet
	 */
	public static int checkLink(int whichDistribution, int whichLink)
	{		
		switch (whichDistribution) 
		{
          case DistributionSettings.NO:
        	  return checkLinkNO(whichLink);
          case DistributionSettings.TF:
        	  return checkLinkTF(whichLink);
          case DistributionSettings.GA:
        	  return checkLinkGA(whichLink);
          case DistributionSettings.GT:
        	  return checkLinkGT(whichLink);
          case DistributionSettings.ST3:
        	  return checkLinkST3(whichLink);
          case DistributionSettings.ST4:
        	  return checkLinkST4(whichLink);
          case DistributionSettings.JSUo:
        	  return checkLinkJSUo(whichLink);
          case DistributionSettings.TF2:
        	  return checkLinkTF2(whichLink);
          case DistributionSettings.SST:
        	  return checkLinkSST(whichLink);
          case DistributionSettings.BCPE:
        	  return checkLinkBCPE(whichLink);
          case DistributionSettings.ST1:
        	  return checkLinkST1(whichLink); 
          case DistributionSettings.PE:
        	  return checkLinkPE(whichLink); 
        	  
        	  
          default: 
			System.err.println("The specific distribution has not been implemented yet!");
			return -1;
         }
	}
	
//-------------------------------------------------------------------------------------------------		
	/**
	 * Checks whether supplied link function is allowed for the NO distribution
	 * @param whichLink - type of link function
	 * @return -1 means that the specific link function is not allowed for NO distribution
	 */
	private static int checkLinkNO(int whichLink)
	{		
		switch (whichLink) 
		{
          case DistributionSettings.IDENTITY:
        	  return DistributionSettings.IDENTITY;
          case DistributionSettings.LOG:
       	     return DistributionSettings.LOG;
          case DistributionSettings.INVERSE:
          	 return DistributionSettings.INVERSE;
          case DistributionSettings.OWN:
        	  return DistributionSettings.OWN;
          default: 
			System.err.println("link function is not supported");
			return -1;
         }
	}	
	
//-------------------------------------------------------------------------------------------------		
	/**
	 * Checks whether supplied link function is allowed for the TF distribution
	 * @param whichLink - type of link function
	 * @return -1 means that the specific link function is not allowed for TF distribution
	 */
	private static int checkLinkTF(int whichLink)
	{		
		switch (whichLink) 
		{
          case DistributionSettings.IDENTITY:
        	  return DistributionSettings.IDENTITY;
          case DistributionSettings.LOG:
       	     return DistributionSettings.LOG;
          case DistributionSettings.INVERSE:
          	 return DistributionSettings.INVERSE;
          case DistributionSettings.OWN:
        	  return DistributionSettings.OWN;
          default: 
			System.err.println("link function is not supported");
			return -1;
         }
	}

//-------------------------------------------------------------------------------------------------		
	/**
	 * Checks whether supplied link function is allowed for the GA distribution
	 * @param whichLink - type of link function
	 * @return -1 means that the specific link function is not allowed for GA distribution
	 */
	private static int checkLinkGA(int whichLink)
	{		
		switch (whichLink) 
		{
        case DistributionSettings.IDENTITY:
      	  return DistributionSettings.IDENTITY;
        case DistributionSettings.LOG:
     	     return DistributionSettings.LOG;
        case DistributionSettings.INVERSE:
        	 return DistributionSettings.INVERSE;
        case DistributionSettings.OWN:
      	  return DistributionSettings.OWN;
        default: 
			System.err.println("link function is not supported");
			return -1;
       }
	}
	

//-------------------------------------------------------------------------------------------------		
	/**
	 * Checks whether supplied link function is allowed for the GT distribution
	 * @param whichLink - type of link function
	 * @return -1 means that the specific link function is not allowed for GT distribution
	 */
	private static int checkLinkGT(int whichLink)
	{		
		switch (whichLink) 
		{
      case DistributionSettings.IDENTITY:
    	  return DistributionSettings.IDENTITY;
      case DistributionSettings.LOG:
   	     return DistributionSettings.LOG;
      case DistributionSettings.INVERSE:
      	 return DistributionSettings.INVERSE;
      case DistributionSettings.OWN:
    	  return DistributionSettings.OWN;
      default: 
			System.err.println("link function is not supported");
			return -1;
     }
	}
	

//-------------------------------------------------------------------------------------------------		
	/**
	 * Checks whether supplied link function is allowed for the ST3 distribution
	 * @param whichLink - type of link function
	 * @return -1 means that the specific link function is not allowed for ST3 distribution
	 */
	private static int checkLinkST3(int whichLink)
	{		
		switch (whichLink) 
		{
    case DistributionSettings.IDENTITY:
  	  return DistributionSettings.IDENTITY;
    case DistributionSettings.LOG:
 	     return DistributionSettings.LOG;
    case DistributionSettings.INVERSE:
    	 return DistributionSettings.INVERSE;
    case DistributionSettings.OWN:
  	  return DistributionSettings.OWN;
    default: 
			System.err.println("link function is not supported");
			return -1;
   }
	}
	
//-------------------------------------------------------------------------------------------------		
	/**
	 * Checks whether supplied link function is allowed for the ST4 distribution
	 * @param whichLink - type of link function
	 * @return -1 means that the specific link function is not allowed for ST4 distribution
	 */
	private static int checkLinkST4(int whichLink)
	{		
		switch (whichLink) 
		{
  case DistributionSettings.IDENTITY:
	  return DistributionSettings.IDENTITY;
  case DistributionSettings.LOG:
	     return DistributionSettings.LOG;
  case DistributionSettings.INVERSE:
  	 return DistributionSettings.INVERSE;
  case DistributionSettings.OWN:
	  return DistributionSettings.OWN;
  default: 
			System.err.println("link function is not supported");
			return -1;
 }
	}

//-------------------------------------------------------------------------------------------------		
	/**
	 * Checks whether supplied link function is allowed for the JSUo distribution
	 * @param whichLink - type of link function
	 * @return -1 means that the specific link function is not allowed for JSUo distribution
	 */
	private static int checkLinkJSUo(int whichLink)
	{		
		switch (whichLink) 
		{
			case DistributionSettings.IDENTITY:
				return DistributionSettings.IDENTITY;
			case DistributionSettings.LOG:
				return DistributionSettings.LOG;
			case DistributionSettings.INVERSE:
				return DistributionSettings.INVERSE;
			case DistributionSettings.OWN:
				return DistributionSettings.OWN;
			default: 
				System.err.println("link function is not supported");
			return -1;
		}
	}

//-------------------------------------------------------------------------------------------------		
	/**
	 * Checks whether supplied link function is allowed for the TF2 distribution
	 * @param whichLink - type of link function
	 * @return -1 means that the specific link function is not allowed for TF2 distribution
	 */
	private static int checkLinkTF2(int whichLink)
	{		
		switch (whichLink) 
		{
			case DistributionSettings.LOGSHIFTTO2:
				return DistributionSettings.LOGSHIFTTO2;
			case DistributionSettings.IDENTITY:
				return DistributionSettings.IDENTITY;
			case DistributionSettings.LOG:
				return DistributionSettings.LOG;
			case DistributionSettings.INVERSE:
				return DistributionSettings.INVERSE;
			case DistributionSettings.OWN:
				return DistributionSettings.OWN;
			default: 
				System.err.println("link function is not supported");
			return -1;
		}
	}

//-------------------------------------------------------------------------------------------------		
	/**
	 * Checks whether supplied link function is allowed for the TF2 distribution
	 * @param whichLink - type of link function
	 * @return -1 means that the specific link function is not allowed for TF2 distribution
	 */
	private static int checkLinkSST(int whichLink)
	{		
		switch (whichLink) 
		{
			case DistributionSettings.LOGSHIFTTO2:
				return DistributionSettings.LOGSHIFTTO2;
			case DistributionSettings.IDENTITY:
				return DistributionSettings.IDENTITY;
			case DistributionSettings.LOG:
				return DistributionSettings.LOG;
			case DistributionSettings.INVERSE:
				return DistributionSettings.INVERSE;
			case DistributionSettings.OWN:
				return DistributionSettings.OWN;
			default: 
				System.err.println("link function is not supported");
			return -1;
		}
	}
//-------------------------------------------------------------------------------------------------		
/**
 * Checks whether supplied link function is allowed for the BCPE distribution
 * @param whichLink - type of link function
 * @return -1 means that the specific link function is not allowed for BCPE distribution
 */
private static int checkLinkBCPE(int whichLink)
{		
	switch (whichLink) 
	{
		case DistributionSettings.IDENTITY:
			return DistributionSettings.IDENTITY;
		case DistributionSettings.LOG:
			return DistributionSettings.LOG;
		case DistributionSettings.INVERSE:
			return DistributionSettings.INVERSE;
		case DistributionSettings.OWN:
			return DistributionSettings.OWN;
		default: 
			System.err.println("link function is not supported");
		return -1;
	}
}

//-------------------------------------------------------------------------------------------------		
/**
* Checks whether supplied link function is allowed for the ST1 distribution
* @param whichLink - type of link function
* @return -1 means that the specific link function is not allowed for BCPE distribution
*/
private static int checkLinkST1(int whichLink)
{		
	switch (whichLink) 
	{
		case DistributionSettings.IDENTITY:
			return DistributionSettings.IDENTITY;
		case DistributionSettings.LOG:
			return DistributionSettings.LOG;
		case DistributionSettings.INVERSE:
			return DistributionSettings.INVERSE;
		case DistributionSettings.OWN:
			return DistributionSettings.OWN;
		default: 
			System.err.println("link function is not supported");
		return -1;
	}
}

//-------------------------------------------------------------------------------------------------		
	/**
	* Checks whether supplied link function is allowed for the PE distribution
	* @param whichLink - type of link function
	* @return -1 means that the specific link function is not allowed for BCPE distribution
	*/
	private static int checkLinkPE(int whichLink)
	{		
		switch (whichLink) 
		{
			case DistributionSettings.IDENTITY:
				return DistributionSettings.IDENTITY;
			case DistributionSettings.LOG:
				return DistributionSettings.LOG;
			case DistributionSettings.INVERSE:
				return DistributionSettings.INVERSE;
			case DistributionSettings.OWN:
				return DistributionSettings.OWN;
			default: 
				System.err.println("link function is not supported");
			return -1;
		}
	}
//-------------------------------------------------------------------------------------------------		
	/**
	 * Returns true if eta satisfies the condition for identity type of link function, else - false
	 * @param eta - vector of linear predictor values
	 * @return boolean
	 */
	public static boolean identityValideta(ArrayRealVector eta){
		return true;
	}

//-------------------------------------------------------------------------------------------------		
	/**
	 * Returns true if eta satisfies the condition for log type of link function, else - false
	 * @param eta - vector of linear predictor values
	 * @return boolean
	 */
	public static boolean logValideta(ArrayRealVector eta){
		return true;
	}

//-------------------------------------------------------------------------------------------------		
	/**
	 * Returns true if eta satisfies the condition for logShiftTo2 type of link function, else - false
	 * @param eta - vector of linear predictor values
	 * @return boolean
	 */
	public static boolean logShiftTo2Valideta(ArrayRealVector eta){
		return true;
	}
//-------------------------------------------------------------------------------------------------	
	/**
	 *  Calculates the values of distParameterEta vector depending on the type of link function
	 * @param whichLink - type of link function
	 * @param eta - vector of linear predictor values
	 * @return distParameterEta vector
	 */
	public ArrayRealVector distParameterEta(int whichLink, ArrayRealVector eta){	
		ArrayRealVector out = null;
		switch (whichLink) 
		{
        case DistributionSettings.IDENTITY:
        	 out = identityDistParameterEta(eta);
           break;
        case DistributionSettings.LOG:
       	     out = logDistParameterEta(eta);
          break;
        case DistributionSettings.LOGSHIFTTO2:
      	     out = logShiftTo2DistParameterEta(eta);
         break;
          
    }
		return out;
	}

//-------------------------------------------------------------------------------------------------	
	/**
	 * Calculates the values of distribution parameter  Eta vector according to identity link function
	 * @param eta - vector of linear predictor values
 	* @return muEta vector
	 */
	public ArrayRealVector identityDistParameterEta(ArrayRealVector eta){
		ArrayRealVector out = new ArrayRealVector(new double[eta.getDimension()], false);		
		out.set(1);
		return out;	
	}
	
//-------------------------------------------------------------------------------------------------
    /**
     * Calculates the values of  distribution parameter  Eta vector according to log link function
     * @param eta - vector of linear predictor values
     * @return muEta vector
     */
	public ArrayRealVector logDistParameterEta(ArrayRealVector eta){
	    int size = eta.getDimension();
	    double[] out = new double[size];
		for (int i=0; i<size; i++)
		{
		    out[i]= FastMath.exp(eta.getEntry(i));
		    if (out[i] < 2.220446e-16) //!!!!    .Machine$double.eps
		    {
		    	out[i] = 2.220446e-16;

		    }			    
		}
		return new ArrayRealVector(out,false);
	}

//-------------------------------------------------------------------------------------------------
/**
 * Calculates the values of distribution parameter Eta vector according to log link function
 * @param eta - vector of linear predictor values
 * @return muEta vector
 */
public ArrayRealVector logShiftTo2DistParameterEta (ArrayRealVector eta){
    int size = eta.getDimension();
    double[] out = new double[size];
	for (int i=0; i<size; i++)
	{
		//mu.eta <- function(eta) pmax(.Machine$double.eps, exp(eta))
	    out[i]= FastMath.exp(eta.getEntry(i));
	    if (out[i] < Double.MIN_VALUE) //!!!!    .Machine$double.eps
	    {
	    	out[i] =Double.MIN_VALUE;
	    }
	}
	return new ArrayRealVector(out,false);
}
	
//-------------------------------------------------------------------------------------------------	
	/**
	 *  Calculates a fitted values of the distribution parameter vector according to the type of link function
	 * @param whichLink - type of link function
	 * @param eta - vector of linear predictor values
	 * @return vector of fitted values 
	 */
	public ArrayRealVector linkInv(int whichLink, ArrayRealVector eta){		
		ArrayRealVector out = null;
		switch (whichLink) 
		{
        case DistributionSettings.IDENTITY:
        	 out = identityInv(eta);
           break;
        case DistributionSettings.LOG:
       	     out = logInv(eta);
          break;
        case DistributionSettings.LOGSHIFTTO2:
      	     out = logShiftTo2Inv(eta);
         break;
    }
		return out;
	}
	
	
//-------------------------------------------------------------------------------------------------	
	/**
	 * 	Calculates a linear predictor values of the distribution parameter vector according to the type of link function
	 * @param whichLink - type of link function
	 * @param whichDistParameter - the distribution parameter
	 * @return vector of linear predictor values 
	 */
	public ArrayRealVector link(int whichLink, ArrayRealVector whichDistParameter){		
			ArrayRealVector out = null;
			switch (whichLink) 
			{
	        case DistributionSettings.IDENTITY:
	        	 out = identity(whichDistParameter);
	           break;
	        case DistributionSettings.LOG:
	       	     out = log(whichDistParameter);
	          break;
	        case DistributionSettings.LOGSHIFTTO2:
	       	     out = logShiftTo2(whichDistParameter);
	          break;
			}
			return out;
		}
	
	
//-------------------------------------------------------------------------------------------------	
	/**
	 *  Calculates a fitted values of the distribution parameter vector according to the identity link function
	 * @param eta - vector of linear predictor values
	 * @return vector of fitted values
	 */
	public ArrayRealVector identityInv(ArrayRealVector eta){	
		return eta;
	}	
	
//-------------------------------------------------------------------------------------------------	
	/**
	 *  Calculates a fitted values of the distribution parameter vector according to the log link function
	 * @param eta - vector of linear predictor values
	 * @return vector of fitted values
	 */
	public ArrayRealVector logInv(ArrayRealVector eta){
	    int size = eta.getDimension();
	    double[] out = new double[size];
		for (int i=0; i<size; i++)
		{
		    out[i]= FastMath.exp(eta.getEntry(i));
		    if (out[i] < 2.220446e-16) //!!!!    .Machine$double.eps
		    {
		    	out[i] = 2.220446e-16;
		    }		    
		}
		return new ArrayRealVector(out,false);	
	}

//-------------------------------------------------------------------------------------------------	
	/**
	 *  Calculates a fitted values of the distribution parameter vector according to the log link function
	 * @param eta - vector of linear predictor values
	 * @return vector of fitted values
	 */
	public ArrayRealVector logShiftTo2Inv(ArrayRealVector eta){
	    int size = eta.getDimension();
	    double[] out = new double[size];
		for (int i=0; i<size; i++)
		{
			//linkinv <- function(eta) 2+pmax(.Machine$double.eps, exp(eta)) 
		    out[i]= FastMath.exp(eta.getEntry(i));
		    if (out[i] < 2.220446e-16) //!!!!    .Machine$double.eps
		    {
		    	out[i] = 2.220446e-16;
		    }
		    out[i] = out[i]+2;
		}
		return new ArrayRealVector(out,false);	
	}
	
//-------------------------------------------------------------------------------------------------	
	/**
	 * 	Calculates a linear predictor values of the distribution parameter vector according to the log link function
	 * @param whichDistParameter - the distribution parameter
	 * @return vector of linear predictor values
	 */
	public ArrayRealVector log(ArrayRealVector whichDistParameter){
	    int size = whichDistParameter.getDimension();
	    double[] out = new double[size];
		for (int i=0; i<size; i++)
		{
		    out[i]= FastMath.log(whichDistParameter.getEntry(i));	
		}
	
		return new ArrayRealVector(out,false);	
	}

//-------------------------------------------------------------------------------------------------	
	/**
	 * 	Calculates a linear predictor values of the distribution parameter vector according to the logshiftto2 link function
	 * @param whichDistParameter - the distribution parameter
	 * @return vector of linear predictor values
	 */
	public ArrayRealVector logShiftTo2(ArrayRealVector whichDistParameter){
	    int size = whichDistParameter.getDimension();
	    double[] out = new double[size];
		for (int i=0; i<size; i++)
		{
			//linkfun <- function(mu)  log(mu-2+0.00001) # changed 6-7-12
		    out[i]= FastMath.log(whichDistParameter.getEntry(i)-2+0.00001);
		}
		return new ArrayRealVector(out,false);	
	}

	
//-------------------------------------------------------------------------------------------------		
	/**
	 * 	Calculates a linear predictor values of the distribution parameter vector according to the identity link function
	 * @param whichDistParameter - the distribution parameter
	 * @return vector of linear predictor values
	 */
	public ArrayRealVector identity(ArrayRealVector whichDistParameter){	
		return whichDistParameter;
	}	
}
