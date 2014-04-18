""" Scraping functions to get images from deviantArt, based on the deviantArt RSS feed.
The functions in this module were only tested in ubuntu 13.10. There is no guarantee
that they run correctly on any other platform. There is no guarantee at all that these
functions are secure, regardless of platform. This is a hack, really :) .
"""
import feedparser as fp
import urllib
import requests
import re
import json
import os.path
import cv2

def fanartScraper(searchString, nbImages, sortBy, outputFolder):
    """ Scrapes fanart images out of deviantArt, writing them to a specific folder.
        Also puts all the metadata into a json file for future usage.
    Args:
        searchString (str): search query to submit to deviantArt.
        nbImages (int): number of images to scrape.
        sortBy (str): string specifying how deviantArt sorts the results. Can be:
            'popular'
        outputFolder (str): folder to write the images and json info files to.
    Raises:
        ValueError: when nbImages is greater than the number of images returned by the
            feed.
        RuntimeError: when the rss feed provides unexpected data.
    """
    # First constructing the proper rss url with proper encoding for everything.
    searchWords = re.sub("[^\w]", " ", urllib.quote_plus(searchString)).split()
    plusedSearchString = reduce(lambda word, rest: word + '+' + rest, searchWords)
    rssUrl = 'http://backend.deviantart.com/rss.xml?type=deviation&q=boost' + urllib.quote_plus(':' + sortBy) + '+' + urllib.quote_plus('in:fanart') + '+' + plusedSearchString
    feed = fp.parse(rssUrl)
    
    # Then scrape each entry individually into image and json files.
    if len(feed.entries) < nbImages:
        raise ValueError("Cannot scrape " + nbImages 
                         + " because the feed only contains " + len(feed.entries))
    # Set the base filename for both json and image files
    baseFilename = reduce(lambda word, rest: word + '_' + rest, searchWords)
    currentImageIdx = 0
    currentEntryIdx = 0

    while currentImageIdx < nbImages and currentEntryIdx < len(feed.entries):
        filename = baseFilename + '_' + repr(currentImageIdx)
        currentEntry = feed.entries[currentEntryIdx]
        # First check that the image url has the right mime type 
        resp = requests.get(currentEntry.media_content[0]['url'])
        fileExtension = None
        contentType = resp.headers['content-type']
        typeToExt = {
            'image/jpeg': 'jpg',
            'image/png': 'png',
            'image/gif': 'gif',
            'image/svg+xml': 'svg'
            }
        try:
            fileExtension = typeToExt[contentType]
        except KeyError:
            raise RuntimeError("The url specified by the field for " + filename 
                               + " is not an image.")
        # Display the image and ask the user whether it's the right character
        # Can't think of an easy way to load the binary data straight into opencv, so
        # I first write it to a file which is then deleted if it's not the right char.
        imageFilename = os.path.join(outputFolder, filename + '.' + fileExtension)
        urllib.urlretrieve(currentEntry.media_content[0]['url'],imageFilename)
        image = cv2.imread(imageFilename)
        pressedKey = -1
        yesCode = ord('y')
        noCode = ord('n')
        while pressedKey != yesCode and pressedKey != noCode:
            cv2.imshow("Is this character " + searchString + "?", image)
            pressedKey = cv2.waitKey(0)
        if pressedKey == yesCode:
            # Write the entire entry to a json file for exhaustiveness
            jsonFile = open(os.path.join(outputFolder, filename + '.json'), 'w')
            json.dump(currentEntry, jsonFile, indent = 4, default = repr)
            jsonFile.close()
            currentImageIdx += 1
        else:
            os.remove(imageFilename)
        currentEntryIdx += 1

if __name__ == "__main__":
    characterNames = [
        'asuka langley',
        'rei ayanami',
        'miku hatsune',
        'monkey d luffy',
        'roronoa zoro',
        'uzumaki naruto',
        'sakura haruno',
        'phoenix wright',
        'maya fey',
        'suzumiya haruhi',
        'asahina mikuru',
        'ginko',
        'yu narukami',
        'naoto shirogane',
        'shigeru akagi',
        'kaiji',
        'lain',
        'faye valentine',
        'radical edward',
        'motoko kusanagi'
        ]
    
    for name in characterNames:
        fanartScraper(name, 10, 'popular', os.path.join('data', 'background'))

