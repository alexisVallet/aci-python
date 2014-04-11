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

    for i in range(0, nbImages):
        filename = baseFilename + '_' + repr(i)
        # Write the entire entry to a json file for exhaustiveness
        jsonFile = open(os.path.join(outputFolder, filename + '.json'), 'w')
        json.dump(feed.entries[i], jsonFile, indent = 4, default = repr)
        jsonFile.close()
        # First check that the image url has the right mime type 
        resp = requests.get(feed.entries[i].media_content[0]['url'])
        fileExtension = None
        contentType = resp.headers['content-type']
        typeToExt = {
            'image/jpeg': 'jpg',
            'image/png': 'png',
            'image/gif': 'gif',
            'image/svg+xml': 'svg'
            }
        fileExtension = typeToExt[contentType]
        # Write the image to the file. urllib provides a method for that so whatever.
        urllib.urlretrieve(feed.entries[i].media_content[0]['url'],
                           os.path.join(outputFolder, filename + '.' + fileExtension))

fanartScraper('asuka langley', 50, 'popular', os.path.join('data', 'background'))
