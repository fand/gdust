#!/usr/bin/env node

var secret = require('./plotly-secret');
var plotly = require('plotly')(secret.name, secret.key);

if (process.argv.length !== 3) {
  console.error('Wrong arguments!');
  console.error(USAGE);
  process.exit();
}
var fs = require('fs');
var json = fs.readFileSync(process.argv[2]);
var data = JSON.parse(json);

// var data = [{
//   "x": [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49],
//   "y": [6,6,6,5,8,6,5,7,5,6,7,5,6,3,6,6,4,6,6,6,7,6,6,4,5,4,6,3,6,3,6,7,6,6,5,6,5,6,6,4,4,6,6,5,4,6,6,6,7,5],"type":"bar","name":"dust"
// },{
//   "x": [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49],
//   "y": [6,6,6,5,8,6,5,7,5,6,7,5,7,3,5,6,4,7,6,6,7,5,7,4,5,4,6,3,6,3,5,7,6,6,5,6,6,3,2,2,1,1,2,2,1,0,0,1,2,0],"type":"bar","name":"gdust"
// },{
//   "x": [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49],
//   "y": [6,6,6,5,8,6,5,7,5,6,7,5,7,3,5,6,4,6,6,6,7,6,6,5,5,4,6,3,6,3,5,7,6,6,5,6,5,2,4,4,4,4,2,2,3,3,4,2,4,4],"type":"bar","name":"gdust_simpson"
// }];
var layout = {
  barmode: "group",
  fileopt : "extend",
  filename : process.argv[2]
};
//var graph_options = {layout: layout};
plotly.plot(data, layout, function (err, msg) {
  if (err) { throw err; }
  console.log(msg);
});
