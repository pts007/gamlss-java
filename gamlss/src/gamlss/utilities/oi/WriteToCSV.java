/*
  Copyright 2012 by Dr. Vlasios Voudouris and ABM Analytics Ltd
  Licensed under the Academic Free License version 3.0
  See the file "LICENSE" for more information
*/
package gamlss.utilities.oi;

import java.io.*;


public final class WriteToCSV {
	
	public  WriteToCSV(double inData) {
		try {
			// Create file 
			FileWriter fstream = new FileWriter("C:/Users/Daniil/Desktop/Gamlss_exp/outTemp.csv", true);
			BufferedWriter out = new BufferedWriter(fstream);
			out.append(',');
			out.write(Double.toString(inData));
			out.append(',');
			out.newLine();
			
			out.close();
		} catch (Exception e) { //Catch exception if any
			System.err.println("Error: " + e.getMessage());
		}
		
	}
	
/*	public void writeToCSV(ACEWEMmodel market, double[] aggregateLMP) {
		try{
			// Create file 
			FileWriter fstream = new FileWriter("results/test2.csv", true);
			BufferedWriter out = new BufferedWriter(fstream);
			
			for (int i = 0; i < market.getGenCoList().size(); i++) {
				GenCo gen = market.getGenCoList().get("genco" + (i + 1));
				for (int j = 0; j < aggregateLMP.length; j++) {
					if (gen.getNode() == j + 1) {
						out.write(Double.toString(aggregateLMP[j] / 24));
						out.append(',');
					}
				}

			}
			
			for (int i = 0; i < market.getGenCoList().size(); i++) {
			out.write(Double.toString(market.getGenCoList().get("genco"
						+ (i + 1)).getReportedSupplyOffer().get("aR")));
			out.append(',');
			}
			
			for (int i = 0; i < market.getGenCoList().size(); i++) {
			out.write(Double.toString(market.getGenCoList().get("genco"
						+ (i + 1)).getReportedSupplyOffer().get("bR")));
			out.append(',');
			}
			for (int i = 0; i < market.getGenCoList().size(); i++) {
			out.write(Double.toString(market.getGenCoList().get("genco"
										  + (i + 1)).getprofitDaily()));
			out.append(',');
			}
			out.newLine();
			//Close the output stream
			out.close();
		} catch (Exception e) { //Catch exception if any
			System.err.println("Error: " + e.getMessage());
		}
	}
	
	public void writeToCSV( Hashtable<Integer, Object> data) {
		try {
			// Create file 
			FileWriter fstream = new FileWriter("results/test2.csv", true);
			BufferedWriter out = new BufferedWriter(fstream);
			
			
			for (int i = 0; i < data.size(); i++) {
				
				 Hashtable<String, String> temp = (Hashtable) data.get(i);
				

				 out.write(temp.get("from"));
				 
				 out.append(',');
				 out.write(temp.get("to"));
				 out.append(',');
				 out.write(temp.get("reactance"));
				 out.append(',');
				 out.write(temp.get("capacity"));
				 out.append(',');
				 out.newLine();
			
//System.out.println(temp.get("from")+"  "+temp.get("to")+"  "
//				 + temp.get("reactance")+"   "+temp.get("capacity"));
			}

			//Close the output stream
			out.close();
		} catch (Exception e) { //Catch exception if any
			System.err.println("Error: " + e.getMessage());
		}
	}
*/	
	
 }