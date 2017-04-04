import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.HashSet;


public class DataPrepare {

	public static void main(String[] args) {
		// TODO Auto-generated method stub
		BufferedReader br = null;
		BufferedWriter bw1 = null;
		BufferedWriter bw2 = null;
		BufferedWriter bw3 = null;

		try {
			
			File multFile = new File("mult.dat");
			File seqFile = new File("seq.dat");
			File dictFile = new File("dict.txt");
			
			if (!multFile.exists()) {
				multFile.createNewFile();
			}
			
			if (!seqFile.exists()) {
				seqFile.createNewFile();
			}
			
			FileWriter fw1 = new FileWriter(multFile.getAbsoluteFile());
			FileWriter fw2 = new FileWriter(seqFile.getAbsoluteFile());
			FileWriter fw3 = new FileWriter(dictFile.getAbsoluteFile());
			
			bw1 = new BufferedWriter(fw1);
			bw2 = new BufferedWriter(fw2);
			bw3 = new BufferedWriter(fw3);

			String sCurrentLine;
			int index = 1;    //index for the hashmap/dictionary
			int uniqueWords = 0;    //count of unique words for a specific doc
			int docCount = 0;
			String time = "0000";
			String seq = "";
			HashMap dict = new HashMap();
			HashMap docWordsCount = new HashMap();

			br = new BufferedReader(new FileReader("data_input_body33.txt"));
			List<String> rawTexts = new ArrayList<String>();
			while ((sCurrentLine = br.readLine()) != null) {
				rawTexts.add(sCurrentLine);
			}
			Collections.sort(rawTexts, new Comparator<String>() {
				@Override
				public int compare(String s1, String s2) {
					String[] s1a = s1.split("\\s+");
					String[] s2a = s2.split("\\s+");
					int time1 = Integer.parseInt(s1a[2]);
					int time2 = Integer.parseInt(s2a[2]);
					return time1-time2;
				}
			});

			time = rawTexts.get(0).split("\\s+")[2];
			
			br = new BufferedReader(new FileReader("en.txt"));
			Set<String> stopWords = new HashSet<String>();
			while ((sCurrentLine = br.readLine()) != null) {
				stopWords.add(sCurrentLine);
			}


			//while ((sCurrentLine = br.readLine()) != null) {
			for(String str: rawTexts) {
				docWordsCount = new HashMap();
				
				//System.out.println(sCurrentLine);
				String[] contents = str.split("\\s+");
				String timeStamp = contents[2];
				if(time.equals(timeStamp)) {
					docCount++;
				} else {
					seq += docCount;
					seq += "\n";
					docCount = 1;
				}
				time = timeStamp;
				for (int i=3;i<contents.length;i++) {
					String word = contents[i].replaceAll("[^a-zA-Z]", "").toLowerCase();
					if (stopWords.contains(word)) {
						continue;
					}

					if (!dict.containsKey(word)) {
						dict.put(word, index++);
					}
					
					if (!docWordsCount.containsKey(word)) {
						int temp = 1;
						docWordsCount.put(word, temp);
					} else {
						int wordCount = (int) docWordsCount.get(word);
						docWordsCount.remove(word);
						wordCount++;
						docWordsCount.put(word, wordCount);
					}
				}
				uniqueWords = docWordsCount.keySet().size();
				
				String content = "";
				content += uniqueWords;
				content += " ";
				Set wordSet = docWordsCount.entrySet();
				Iterator i = wordSet.iterator();
				while (i.hasNext()) {
					Map.Entry me = (Map.Entry)i.next();
					int currIndex = (int) dict.get(me.getKey());
					int currCount = (int) me.getValue();
					content += (currIndex+":"+currCount);
					content += " ";
				}
				content += "\n";
				bw1.write(content);
				
			}

			Set dictSet = dict.entrySet();
			Iterator iter = dictSet.iterator();
			while (iter.hasNext()) {
				Map.Entry entry = (Map.Entry) iter.next();
				String word = (String) entry.getKey();
				int dictIndex = (int) entry.getValue();
				String dictEntry = dictIndex + "," + word + "\n";
				bw3.write(dictEntry);
			}
			seq += docCount;
			seq += "\n";
			int numOfLines = countLines(seq);
			seq = numOfLines + "\n" + seq;
			
			bw2.write(seq);
			


		} catch (IOException e) {
			e.printStackTrace();
		} finally {
			try {
				if (br != null)br.close();
				bw1.close();
				bw2.close();
				bw3.close();
			} catch (IOException ex) {
				ex.printStackTrace();
			}
		}


	}
	
//	private static int countLines(String str) {
//		if(str == null || str.isEmpty())
//	    {
//	        return 0;
//	    }
//	    int lines = 1;
//	    int pos = 0;
//	    while ((pos = str.indexOf("\n", pos) + 1) != 0) {
//	        lines++;
//	    }
//	    return lines;
//	}
	
	private static int countLines(String str){
	   String[] lines = str.split("\r\n|\r|\n");
	   return  lines.length;
	}

}
