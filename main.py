from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

load_dotenv()


def main() -> None:
    information = """
    From Wikipedia, the free encyclopedia
    This article is about the empire. For the associated caliphate, see Ottoman Caliphate.
    "The Ottoman Empire" redirects here. For the band formerly known under that name, see Luna Mortis.
    Sublime Ottoman State
    دولت عليه عثمانیه
    Devlet-i ʿAlīye-i ʿOs̱mānīye
    c. 1299–1922
    Flag of Ottoman Empire
    Flag
    (1844–1922)
    Coat of arms(1882–1922) of Ottoman Empire
    Coat of arms
    (1882–1922)
    Motto: 
    دولت ابدمدت
    Devlet-i Ebed-müddet
    "The Eternal State"[1]
    Anthem: 
    Various

    The Ottoman Empire in 1481
    The Ottoman Empire in 1566
    The Ottoman Empire in 1683
    The Ottoman Empire in 1739
    The Ottoman Empire in 1914
    Show all
    Status	Empire
    Capital	
    Söğüt[2]
    (c. 1299–1331)
    Nicaea (İznik)[3]
    (1331–1335)
    Prousa (Bursa)
    (1335–1360s)
    Adrianople (Edirne)[a]
    (1360s–1453)
    Constantinople (Istanbul)[b]
    (1453–1922)
    Official languages	Ottoman Turkish
    Other languages	
    Arabic[c]
    Persian[d]
    Greek[e]
    Chagatai[f]
    French[g]
    many others
    Religion	
    Sunni Islam (state)
    School: Hanafi
    Demonym	Ottoman
    Government	Absolute monarchy (c. 1299–1876; 1878–1908; 1920–1922)
    Parliamentary constitutional monarchy (1876–1878; 1908–1920)
    Under a triumvirate dictatorship (1913–1918)
    Sultan	 
    • c. 1299–1323/4 (first)
    Osman I
    • 1918–1922 (last)
    Mehmed VI
    Caliph	 
    • 1517–1520 (first)
    Selim I[19][h]
    • 1922–1924 (last)
    Abdülmecid II[i]
    Grand Vizier	 
    • 1320–1331 (first)
    Alaeddin Pasha
    • 1920–1922 (last)
    Ahmet Tevfik Pasha
    Legislature	General Assembly
    (1876–1878; 1908–1920)
    • Upper house (unelected)
    Chamber of Notables
    (1876–1878; 1908–1920)
    • Lower house (elected)
    Chamber of Deputies
    (1876–1878; 1908–1920)
    History	 
    • Founded
    c. 1299[20]
    • Interregnum
    1402–1413
    • Conquest of Constantinople
    29 May 1453
    • Proclamation of the Ottoman Caliphate
    1517
    • Constitutional Era I
    1876–1878
    • Constitutional Era II
    1908–1920
    • Ottoman coup d'état
    23 January 1913
    • Sultanate abolished
    1 November 1922[j]
    • Republic of Turkey established
    29 October 1923[k]
    • Caliphate abolished
    3 March 1924
    Area
    1481[21]	1,220,000 km2 (470,000 sq mi)
    1521[21]	3,400,000 km2 (1,300,000 sq mi)
    1683[21][22][23]	5,200,000 km2 (2,000,000 sq mi)
    1913[21]	2,550,000 km2 (980,000 sq mi)
    Population
    • 1600
    22,000,000[24]
    • 1912
    24,000,000[25]
    Currency	Akçe, manghir, sultani, para, kuruş, lira
    Dependencies
    Predecessor states and successor states
    The Ottoman Empire,[l] historically also known as the Turkish Empire or Turkey,[26][27][m] was a state that spanned much of Southeastern Europe, West Asia, and North Africa from the 14th century to the early 20th century, centred in modern-day Turkey. It also controlled parts of southeastern Central Europe between the early 16th and early 18th centuries.[28][29][30]

    The empire emerged from a beylik, or principality, founded in northwestern Anatolia in c. 1299 by the Turkoman tribal leader Osman I. His successors conquered much of Anatolia and expanded into the Balkans by the mid-14th century, transforming their petty kingdom into a transcontinental empire. The Ottomans ended the Byzantine Empire with the conquest of Constantinople in 1453 by Mehmed II. Further conquests by Selim I led the Sultans to adopt the title of caliph. With its capital at Constantinople and control over a significant portion of the Mediterranean Basin, the Ottoman Empire was at the centre of interactions between the Middle East and Europe for six centuries. Ruling over diverse peoples, the empire granted varying levels of autonomy to its many confessional communities, or millets, to manage their own affairs per Islamic law. During the reign of Suleiman the Magnificent in the 16th century, the Ottoman Empire became a global power.[31]

    Modern academic consensus posits that the empire began to decline after defeat in the Second Siege of Vienna in 1683, but continued to maintain a flexible and strong economy, society and military into much of the 18th century. The Ottoman Empire fell behind technologically from the rest of Europe by the late 18th century as imperial authority fragmented. Further defeats from Austria and Russia culminated in the loss of territory, and with rising nationalism after the French Revolution, a number of new states emerged in the Balkans. Following Mahmud II's reign and the Tanzimat reforms over the course of the 19th century, the Ottoman state became more powerful and organised internally as a new Ottoman identity took hold. In the 1876 revolution, the Ottoman Empire attempted constitutional monarchy, before reverting to an absolute monarchy under Abdul Hamid II, following the Great Eastern Crisis.

    Over the course of the late 19th century, Ottoman intellectuals known as Young Turks sought to liberalise and rationalise society and politics along Western lines, culminating in the Young Turk Revolution of 1908 led by the Committee of Union and Progress (CUP), which reestablished a constitutional monarchy. However, following the disastrous Balkan Wars, the CUP became increasingly radicalised and embraced Turkish nationalism, leading to a coup d'état in 1913 that established a dictatorship.

    In the 19th and in the start of 20th centuries, persecution of Muslims during the Ottoman contraction and in the Russian Empire resulted in large-scale loss of life and mass migration into modern-day Turkey from the Balkans, Caucasus, and Crimea.[32] The CUP joined World War I on the side of the Central Powers. It struggled with internal dissent, especially the Arab Revolt, and engaged in genocide against Armenians, Assyrians, and Greeks. In the aftermath of World War I, the victorious Allied Powers occupied and partitioned the Ottoman Empire, which lost its southern territories to the United Kingdom and France. The successful Turkish War of Independence, led by Mustafa Kemal Atatürk against the occupying Allies, led to the end of the Ottoman sultanate in 1922.
    """
    summary_prompt = """ 
    Given the following information {information}: please summerize the following, then give two interesting facts about the text.
    """
    summary_prompt_template = PromptTemplate(
        input_variables=["information"], template=summary_prompt
    )

    llm = ChatOpenAI(temperature=0, model="gpt-5")

    chain = summary_prompt_template | llm
    response = chain.invoke(input={"information": information})

    print(response.content)


if __name__ == "__main__":
    main()
