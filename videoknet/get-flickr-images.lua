
-- Libs
require 'pl'
text.format_operator()
local json = require 'cjson' --dkjson
local unicode = require 'unicode'
local socket = require('socket')
local http = require('socket.http')
local url = require('socket.url')
local ltn12 = require('ltn12')

-- A function that downloads images for a set of tags:
local function getImagesForTags(...)
   -- Tags:
   local tags = table.concat({...},',')

   -- Output dir?
   local odir = path.join('flickr',tags)
   dir.makepath(odir)

   -- Escape:
   tags = tags:gsub(' ','%%20')

   -- URL?
   local query = {
       'http://api.flickr.com/services/rest/',
       {
          'method=flickr.photos.search',
          'api_key=174c6e00391d2b69ce608aa750fcb0e4',
          'text='..tags,
          'format=json',
          'nojsoncallback=1'
       }
   }

   query[2] = table.concat(query[2],'&')
   query = table.concat(query,'?')

   -- Get data 
   local t = {}
   local respt = http.request{
      url = query,
      sink = ltn12.sink.table(t)
   }

   -- Data
   local data = table.concat(t)

   -- Json
   local result = json.decode(data)

   -- Produce URL
   local function url(photo)
      local url = 
         'http://farm${farm}.staticflickr.com/${server}/${id}_${secret}_c.jpg' % photo
      return url
   end

   -- Get image:
   local photos = result.photos.photo
   for i,photo in ipairs(photos) do
      local url = url(photo)
      local fname = path.join(odir,path.basename(url))
      print(url .. ' ==> ' .. fname)
      http.request{
         url = url,
         sink = ltn12.sink.file(io.open(fname,'w+b'))
      }
   end
end

-- A few tests:
getImagesForTags('Los Angeles')
getImagesForTags('Paris', 'Tour Eiffel')
getImagesForTags('New York', 'Broadway')
getImagesForTags('New York', 'Wall Street')
getImagesForTags('Paris','Skate')

