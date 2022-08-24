import logging
import random
from typing import List, Dict

import spacy
from spacy.training.example import Example

from classification.preprocessing import Website
from extraction import nerHelper
from .base_extraction_network import BaseExtractionNetwork


class ExtractionNetworkNerV1(BaseExtractionNetwork):

    model: spacy.Language
    log = logging.getLogger('ExtNet-NerV1')

    def __init__(self, name: str, **kwargs):
        super().__init__(name=name, version='NerV1',
                         description='Try to extract information with custom SPACY-Model')
        self.nlp = spacy.load('en_core_web_md')
        self.EMB_DIM = self.nlp.vocab.vectors_length
        self.MAX_LEN = 50

    def predict(self, web_ids: List[str], **kwargs) -> List[Dict[str, List[str]]]:
        self.load()

        result_list = []
        for web_id in web_ids:
            html_text = nerHelper.get_html_text(web_id)

            website = Website.load(web_id)
            attributes = website.truth.attributes
            attributes.pop('category')

            id_results = {}
            for attr in attributes:
                id_results[str(attr).upper()] = []

            # doc = self.model('\n'.join(html_text))
            doc = self.model(html_text)
            for ent in doc.ents:
                id_results[ent.label_].append(ent.text)

            for label in id_results:
                id_results[label] = [max(set(id_results[label]), key=id_results[label].count)]

            result_list.append(id_results)

        return result_list

    def train(self, web_ids: List[str], **kwargs) -> None:
        epochs = 50

        training_data = {'classes': [], 'annotations': []}
        for web_id in web_ids:
            html_text = nerHelper.get_html_text(web_id)

            website = Website.load(web_id)
            attributes = website.truth.attributes
            attributes.pop('category')
            new_attributes = {}
            for attr, value in attributes.items():
                if value:
                    value_preprocessed = str(value[0]).replace('&nbsp;', ' ').strip()
                    new_attributes[str(attr).upper()] = value_preprocessed
                else:
                    new_attributes[str(attr).upper()] = []

            training_data['annotations'].append(nerHelper.html_text_to_spacy(html_text, new_attributes))

        nlp = spacy.blank("en")  # load a new spacy model

        if 'ner' not in nlp.pipe_names:
            ner = nlp.add_pipe('ner')
        else:
            ner = nlp.get_pipe('ner')

        for attr in attributes:
            training_data['classes'].append(str(attr).upper())
            self.log.debug("Attribute added: %s", attr.upper())
            ner.add_label(attr)

        optimizer = nlp.begin_training()

        other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']
        with nlp.disable_pipes(*other_pipes):  # only train NER
            for itn in range(epochs):
                random.shuffle(training_data['annotations'])
                losses = {}
                for batch in spacy.util.minibatch(training_data['annotations'], size=8):
                    for content in batch:
                        # create Example
                        doc = nlp.make_doc(content['text'])
                        example = Example.from_dict(doc, content)
                        # Update the model
                        nlp.update([example], losses=losses, drop=0.3, sgd=optimizer)
                self.log.info('Iteration %s / %s with loss: %s', itn+1, epochs, losses)

        self.model = nlp
        self.save()

        # Test the trained model
        self.load()
        test_text = 'Steve Blake From Wikipedia, the free encyclopedia Jump to: navigation , search Steve Blake No. 5 Los Angeles Lakers Point guard Personal information Date of birth February 26, 1980 ( 1980-02-26 ) (age 30) Place of birth Hollywood, Florida Nationality American High school Miami Oak Hill Academy Listed height 6 ft 3 in (1.91 m) Listed weight 172 lb (78 kg) Career information College Maryland NBA Draft 2003 / Round: 2 / Pick: 38 Selected by the Washington Wizards Pro career 2003–present Career history Washington Wizards (2003–05) Portland Trail Blazers (2005–06; 2007-10) Milwaukee Bucks (2006–07) Denver Nuggets (2007) Los Angeles Clippers (2010) Los Angeles Lakers (2010-present) Steve Blake at NBA.com Steven Hanson Blake (born February 26, 1980 in Hollywood , Florida ) is an American professional basketball point guard for the Los Angeles Lakers . Previously, he played for the Washington Wizards , Milwaukee Bucks , Denver Nuggets , Portland Trail Blazers and the Los Angeles Clippers . Contents 1 Amateur career 1.1 High school 1.2 College 2 NBA career 2.1 Washington 2.2 Portland 2.3 Milwaukee 2.4 Denver 2.5 Portland 2.6 Los Angeles Clippers 2.7 Los Angeles Lakers 3 NBA career statistics 3.1 Regular season 3.2 Playoffs 4 See also 5 References 6 External links [ edit ] Amateur career [ edit ] High school Blake spent his freshman and sophomore year at Miami Killian High School and then transferred to Miami High School , where he played with another future NBA player, Udonis Haslem . Miami won consecutive state championships, but after the Miami New Times exposed the fact that Blake and other players were using fake addresses to enroll in the school, one of the titles was expunged. [ 1 ] [ 2 ] After being banned from playing again anywhere in the FHSAA, he then attended Oak Hill Academy before coming to Maryland. [ edit ] College After high school, he attended the University of Maryland . Blake was the team\'s starter the first day of his freshman year and was the first ACC player to compile 1,000 points, 800 assists, 400 rebounds and 200 steals. He finished his career 5th in NCAA all-time career assists with 972. Widely known for his superb passing ability, Blake helped lead the Terrapins to a Final Four appearance (2001) and the 2002 NCAA championship; less well-known for his scoring, Blake did average eleven points per game in his senior year. [ 3 ] He averaged over six assists per game in each of his four years, including averages of 7.9 and 7.1 in 2002 and 2003, respectively. For his efforts, he was named to a variety of all-ACC teams during his career, including the rookie and defensive squads, capped by a first-team All-ACC spot his senior year. At the start of the 2003–04 basketball season, Blake\'s uniform number (25) became only the 15th to be retired to the rafters of Maryland\'s Comcast Center. [ 4 ] [ edit ] NBA career [ edit ] Washington Blake was selected by the Washington Wizards with the 38th pick in the 2003 NBA Draft . He averaged 5.9 points, 2.8 assists, and 18.6 minutes per game while playing in 75 games his rookie season with the Wizards. In his second season Blake\'s playing time decreased to 14.7 minutes and only 44 games played. [ edit ] Portland In September 2005, Blake (then a restricted free agent with the Wizards) was offered a contract by the Portland Trail Blazers , which the Wizards declined to match. This became the second reunion with former Maryland Terrapin and Washington Wizards backcourt teammate Juan Dixon , who also signed with the Trail Blazers in the 2005 off-season. In Blake\'s first season with the Blazers, he became a starter and played a significant role when Sebastian Telfair was injured. Blake\'s playing time increased from 14.7 minutes and 44 games with only one start in 2004–05 to 26.2 and 68 games with 57 starts in 05–06 . Blake reestablished himself as a terrific passer and fundamental point guard claiming third in the NBA in assist-to-turnover ratio. He also increased his field goal percentage by 11%. [ edit ] Milwaukee In July 2006, Blake was traded (along with Brian Skinner and Ha Seung-Jin ) to the Milwaukee Bucks for Jamaal Magloire . [ 5 ] [ edit ] Denver On January 11, 2007, Blake was traded to the Denver Nuggets in return for Earl Boykins and Julius Hodge . [ 6 ] Blake started in 40 of the 49 remaining games of the Nuggets\' 2006–07 season, and in five playoff games in a 4–1 first-round loss to the San Antonio Spurs . [ edit ] Portland Blake became an unrestricted free agent on July 1, 2007, and agreed to a three-year deal with the Portland Trail Blazers on July 13, 2007. [ 7 ] The 2008–09 season saw a rise in Blake\'s numbers. Through his first 38 games, he averaged a career-high 11.7 points per game, while also achieving career highs in free throw percentage and three-point percentage. [ 8 ] On February 22, 2009, Blake tied an NBA record with 14 assists in the first quarter of a game against the Los Angeles Clippers . [ 9 ] [ edit ] Los Angeles Clippers On February 16, 2010, Blake was traded to the Los Angeles Clippers with Travis Outlaw and $1.5 million in cash for Marcus Camby . [ 10 ] [ edit ] Los Angeles Lakers On July 8, 2010, Blake officially signed a four-year $16 million contract with the Los Angeles Lakers . [ 11 ] On October 26, 2010, in the season opener against the Houston Rockets , he hit a clutch 3-pointer to give the Lakers the lead. He also defended Houston\'s Aaron Brooks on the following possession, forcing him to miss a tough shot at the buzzer, sealing the win for the Lakers. [ edit ] NBA career statistics Legend GP Games played GS Games started MPG Minutes per game FG% Field-goal percentage 3P% 3-point field-goal percentage FT% Free-throw percentage RPG Rebounds per game APG Assists per game SPG Steals per game BPG Blocks per game PPG Points per game Bold Career high [ edit ] Regular season Year Team GP GS MPG FG% 3P% FT% RPG APG SPG BPG PPG 2003–04 Washington 75 14 18.6 .386 .371 .821 1.6 2.8 .8 .1 5.9 2004–05 Washington 44 1 14.7 .328 .387 .805 1.6 1.6 .3 .0 4.3 2005–06 Portland 68 57 26.2 .438 .413 .791 2.1 4.5 .6 .1 8.2 2006–07 Milwaukee 33 2 17.7 .349 .279 .550 1.4 2.5 .3 .1 3.6 2006–07 Denver 49 40 33.5 .432 .343 .727 2.5 6.6 1.0 .1 8.3 2007–08 Portland 81 78 29.9 .408 .406 .766 2.4 5.1 .7 .1 8.5 2008–09 Portland 69 69 31.7 .428 .427 .840 2.5 5.0 1.0 .1 11.0 2009–10 Portland 51 28 27.4 .403 .377 .750 2.3 4.0 .7 .0 7.6 2009–10 L.A. Clippers 29 10 26.3 .443 .437 .750 2.4 6.1 .7 .1 6.8 Career 499 299 25.7 .411 .393 .780 2.1 4.3 .7 .1 7.5 [ edit ] Playoffs Year Team GP GS MPG FG% 3P% FT% RPG APG SPG BPG PPG 2004–05 Washington 4 0 4.3 .250 .000 .000 .8 .5 .0 .0 .5 2006–07 Denver 5 5 36.0 .452 .500 .000 2.4 4.6 .6 .0 7.2 2008–09 Portland 6 6 38.5 .489 .417 .714 4.0 6.2 .8 .0 9.8 Career 15 11 28.5 .463 .439 .714 2.6 4.1 .5 .0 6.5 [ edit ] See also List of NCAA Division I men\'s basketball career assists leaders [ edit ] References ^ Powell, Robert Andrew (February 24, 2005). "Sanitized by the Herald" . Miami New Times . http://www.miaminewtimes.com/2005-02-24/news/sanitized-by-the-herald/ . Retrieved 2009-04-04 . ^ Watford, Jack (1998-08-11). "Miami High School found guilty of FHSAA rules violations" . FHSAA . http://www.fhsaa.org/news/1998/0811.htm . ^ "Steve Blake, Maryland, 6-3, G" . SportsStats . http://www.sportsstats.com/jazzyj/greats/03/blake.htm . Retrieved 2008-11-29 . [ dead link ] ^ "Athletics - The University of Maryland Terrapins - Official Athletic Site" . UMTerps.com . http://umterps.cstv.com/ot/md-ask-testudo.html . Retrieved 2008-11-29 . ^ "Trail Blazers Acquire All-Star Center Magloire" . NBA.com . 2006-07-31 . http://www.nba.com/blazers/news/Jamaal_Magliore-185803-1218.html . ^ Stein, Marc (2007-01-12). "Nuggets deal Boykins, Hodge to Bucks for Blake" . ESPN . http://sports.espn.go.com/nba/news/story?id=2727760 . ^ Lopez, Aaron J. (July 13, 2007). "Nuggets lose Blake" . Rocky Mountain News . Archived from the original on 2007-07-15 . http://web.archive.org/web/20070715121821/http://www.rockymountainnews.com/drmn/nba/article/0,2777,DRMN_23922_5628146,00.html . Retrieved 2007-07-13 . ^ "Steve Blake 2008-09 stats" . NBA.com . http://www.nba.com/playerfile/steve_blake/career_stats.html . Retrieved January 21, 2009 . ^ "Blake, Trail Blazers top Clippers 116-87" . Associated Press . http://www.google.com/hostednews/ap/article/ALeqM5hapM0xex16cNEA_Kk_J4zidCxH6AD96GVVV80 . Retrieved 2009-02-23 . ^ Quick, Jason (February 16, 2010). "Blazers-Clipper trade: Deal for Marcus Camby completed, Kevin Pritchard says" . The Oregonian . http://blog.oregonlive.com/behindblazersbeat/2010/02/pritchard_trade_for_marcus_cam.html . ^ "Lakers Sign Steve Blake" . NBA.com . 2010-07-08 . http://www.nba.com/lakers/news/100708lakerssignsteveblake.html . [ edit ] External links Official website Steve Blake Info Page at NBA.com Steve Blake NBA & ABA Statistics at Basketball-Reference.com v • d • e Maryland Terrapins Men\'s Basketball 2001–2002 NCAA Champions 1 Byron Mouton | 3 Juan Dixon ( MOP ) | 12 Drew Nicholas | 21 Mike Grinnon | 25 Steve Blake | 33 Ryan Randle | 35 Lonny Baxter | 45 Tahj Holden | 54 Chris Wilcox Coach Gary Williams v • d • e 2003 NBA Draft First round LeBron James · Darko Miličić · Carmelo Anthony · Chris Bosh · Dwyane Wade · Chris Kaman · Kirk Hinrich · T. J. Ford · Michael Sweetney · Jarvis Hayes · Mickaël Piétrus · Nick Collison · Marcus Banks · Luke Ridnour · Reece Gaines · Troy Bell · Žarko Čabarkapa · David West · Aleksandar Pavlović · Dahntay Jones · Boris Diaw · Zoran Planinić · Travis Outlaw · Brian Cook · Carlos Delfino · Ndudi Ebi · Kendrick Perkins · Leandro Barbosa · Josh Howard Second round Maciej Lampe · Jason Kapono · Luke Walton · Jerome Beasley · Sofoklis Schortsanitis · Szymon Szewczyk · Mario Austin · Travis Hansen · Steve Blake · Slavko Vraneš · Derrick Zimmerman · Willie Green · Zaza Pachulia · Keith Bogans · Malick Badiane · Matt Bonner · Sani Bečirovič · Mo Williams · James Lang · James Jones · Paccelis Morlende · Kyle Korver · Remon van de Hare · Tommy Smith · Nedžad Sinanović · Rick Rickert · Brandon Hunter · Xue Yuyang · Andreas Glyniadakis v • d • e Los Angeles Lakers current roster 2 Fisher | 3 Ebanks | 4 Walton | 5 Blake | 7 Odom | 9 Barnes | 12 Brown | 15 Artest | 16 Gasol | 17 Bynum | 18 Vujačić | 24 Bryant | 45 Caracter | 50 Ratliff Head coach: Jackson | Assistant coaches: Hamblen | Shaw | Abdul-Jabbar | Hodges | Person | Cleamons Persondata Name Steve Blake Alternative names Steven Hanson Blake Short description National Basketball Association player. Date of birth February 26, 1979 Place of birth Hollywood , Florida ) Date of death Place of death Retrieved from " http://en.wikipedia.org/wiki/Steve_Blake " Categories : 1980 births | Living people | American basketball players | Basketball players from Florida | Denver Nuggets players | Los Angeles Clippers players | Los Angeles Lakers players | Maryland Terrapins men\'s basketball players | Milwaukee Bucks players | People from Hollywood, Florida | People from West Linn, Oregon | Point guards | Portland Trail Blazers players | Washington Wizards draft picks | Washington Wizards players Hidden categories: All articles with dead external links | Articles with dead external links from September 2010 Personal tools Log in / create account Namespaces Article Discussion Variants Views Read Edit View history Actions Search Navigation Main page Contents Featured content Current events Random article Donate to Wikipedia Interaction Help About Wikipedia Community portal Recent changes Contact Wikipedia Toolbox What links here Related changes Upload file Special pages Permanent link Cite this page Print/export Create a book Download as PDF Printable version Languages Español Français Galego Italiano עברית 日本語 Português Русский This page was last modified on 29 November 2010 at 01:00. Text is available under the Creative Commons Attribution-ShareAlike License ; additional terms may apply. See Terms of Use for details. Wikipedia® is a registered trademark of the Wikimedia Foundation, Inc. , a non-profit organization. Contact us Privacy policy About Wikipedia Disclaimers '
        doc = self.model(test_text)
        print("Entities in '%s'" % test_text)
        for ent in doc.ents:
            print(ent.label_, ent.text)
        # --------

    def load(self) -> None:
        if not self.dir_path.exists():
            raise ValueError(f"The model '{self.name}' for version {self.version} doesn't exit.")
        self.model = spacy.load(self.dir_path)

    def save(self) -> None:
        if self.model is None:
            raise ValueError(f"No model to save. Model '{self.name}' for version {self.version} not set.")
        self.model.meta['name'] = self.name
        self.model.to_disk(self.dir_path.as_posix())
