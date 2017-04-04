import java.io.File;
import java.io.FileWriter;
import java.io.FilenameFilter;
import java.io.IOException;

import javax.xml.parsers.DocumentBuilder;
import javax.xml.parsers.DocumentBuilderFactory;

import org.w3c.dom.Document;
import org.w3c.dom.Element;
import org.w3c.dom.Node;
import org.w3c.dom.NodeList;

public class XmlParser2 {
	public static void main(String[] args) {
		try {
			//String[] requestedPart = {"title","abstract", ""};
			File inputFile = new File("fooName.xml");
			String path = System.getProperty("user.dir");
			System.out.println("Path is: " + path);
			File folder = new File(path);
			File[] listOfFiles = folder.listFiles(new FilenameFilter() {
				public boolean accept(File dir, String filename) {
					return filename.endsWith(".xml");
				}
			});

			FileWriter writer = new FileWriter("data_input_body55.txt");

			DocumentBuilderFactory dbFactory
			= DocumentBuilderFactory.newInstance();
			DocumentBuilder dBuilder = dbFactory.newDocumentBuilder();

			int counter = 1;

			for(int i=0;i<listOfFiles.length; i++) {
				inputFile = new File(listOfFiles[i].getName());
				Document doc = dBuilder.parse(inputFile);
				doc.getDocumentElement().normalize();
				System.out.println("Root element :"
					+ doc.getDocumentElement().getNodeName());
				NodeList nList = doc.getElementsByTagName("article_rec");
				for (int j=0;j<nList.getLength();j++) {
					Node nNode = nList.item(j);
					if (nNode.getNodeType() == Node.ELEMENT_NODE) {
						Element eElement = (Element) nNode;
						NodeList titleList = eElement.getElementsByTagName("title");
						NodeList abstractList = eElement.getElementsByTagName("par");
						NodeList referenceList = eElement.getElementsByTagName("ref_text");
						NodeList citationList = eElement.getElementsByTagName("cited_by_text");
						NodeList bodyList = eElement.getElementsByTagName("ft_body");
						NodeList timeStampList = eElement.getElementsByTagName("article_publication_date");
						String contentString = "";
						// for (int m=0;m<titleList.getLength();m++) {
						// 	Node aNode = titleList.item(m);
						// 	String titleString = aNode.getTextContent();
						// 	titleString = titleString.replaceAll("<.*?>", "").replaceAll("\"", "");
						// 	contentString += titleString;
						// 	contentString += " ";
						// }
						// for (int m=0;m<abstractList.getLength();m++) {
						// 	Node aNode = abstractList.item(m);
						// 	String abstractString = aNode.getTextContent();
						// 	abstractString = abstractString.replaceAll("<.*?>", "").replaceAll("\"", "");
						// 	contentString += abstractString;
						// 	contentString += " ";
						// }
						for (int m=0;m<bodyList.getLength();m++) {
							Node aNode = bodyList.item(m);
							String bodyString = aNode.getTextContent();
							bodyString = bodyString.replaceAll("<.*?>", "").replaceAll("\"", "");
							bodyString = bodyString.replace("\n", "");
							contentString += bodyString;
							contentString += " ";
						}
						for (int m=0;m<timeStampList.getLength();m++) {
							Node aNode = timeStampList.item(m);
							String timeStamp = aNode.getTextContent();
							String[] timeList = timeStamp.split("-");
							timeStamp = timeList[timeList.length-1];
							contentString = timeStamp + " " + contentString;
						}
						// for (int m=0;m<referenceList.getLength();m++) {
						// 	Node aNode = referenceList.item(m);
						// 	contentString += processReference(aNode.getTextContent().replaceAll("\"", ""));
						// 	contentString += " ";
						// }
						// for (int m=0;m<citationList.getLength();m++) {
						// 	Node cNode = citationList.item(m);
						// 	contentString += processReference(cNode.getTextContent().replaceAll("\"", ""));
						// 	contentString += " ";
						// }
						
						contentString = contentString.trim();
						if (contentString.split("\\s+").length <= 20) {
							continue;
						}
						if (!contentString.equals("")) {
							writer.append(Integer.toString(counter++));
							writer.append(" en ");
							writer.append(contentString);
							writer.append('\n');
						}
						
					}
				}
				
			}

			writer.flush();
			writer.close();
		} catch(IOException e) {
			System.out.println("Error writing to csv file. " + e.getMessage());
		} catch(Exception e) {
			System.out.println("Error parsing XML; " + e.getMessage());
			e.printStackTrace();
		}
	}
	
	private static String processReference(String rawString) {
		String[] stringList = rawString.split("\\.");
		for (String stringPart: stringList) {
			stringPart = stringPart.trim();
			String[] innerList = stringPart.split("\\s+");
			if (innerList.length >= 5) {
				return stringPart;
			}
		}
		return "";
	}
}
