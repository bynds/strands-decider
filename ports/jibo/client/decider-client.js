/* A client for `jibo-decider serve`, for skills running on the robot's Node.js.
 *
 * Plain ES5 with callbacks and only the built-in `net` module, because the robot's Node version
 * is old and unknown here. The request and response are those of POST /v1/systemone.
 *
 *   var decider = require('./decider-client');
 *   var client = decider.connect('/path/to/decider.sock', {timeoutMs: 10000});
 *   client.ask({
 *     state: 'Jibo, can you set a timer for ten minutes?',
 *     questions: {
 *       addressed: {type: 'noul', instructions: 'Is the speaker talking to the robot?'},
 *       intent: {type: 'choice', instructions: 'What does the speaker want?',
 *                criteria: {timer: 'set a timer', weather: 'weather', chat: 'small talk'}}
 *     }
 *   }, function (err, response) {
 *     if (err) return console.error(err.message);  // includes "busy: ..." refusals
 *     console.log(response.answers.addressed.noul, response.answers.intent.choice);
 *   });
 *
 * A decision takes seconds on the robot. The timeout only stops waiting: the server finishes the
 * request it is working on, so a skill should not resend at once.
 */
'use strict';

var net = require('net');

function request(socketPath, payload, timeoutMs, callback) {
  var done = false;
  var chunks = [];
  var sock = net.connect(socketPath);

  function finish(err, value) {
    if (done) return;
    done = true;
    clearTimeout(timer);
    sock.destroy();
    callback(err, value);
  }

  var timer = setTimeout(function () {
    finish(new Error('jibo-decider did not answer within ' + timeoutMs + ' ms'));
  }, timeoutMs);

  sock.on('error', function (err) { finish(err); });
  sock.on('data', function (chunk) { chunks.push(chunk); });
  sock.on('end', function () {
    var text = Buffer.concat(chunks).toString('utf8');
    var body;
    try {
      body = JSON.parse(text);
    } catch (e) {
      return finish(new Error('unreadable response from jibo-decider: ' + text.slice(0, 200)));
    }
    if (body && typeof body.detail === 'string') return finish(new Error(body.detail));
    finish(null, body);
  });
  /* write the request, then close our side: the server reads to end of input */
  sock.end(JSON.stringify(payload));
}

function connect(socketPath, options) {
  var timeoutMs = (options && options.timeoutMs) || 30000;
  return {
    ask: function (req, callback) { request(socketPath, req, timeoutMs, callback); },
    health: function (callback) { request(socketPath, {health: true}, timeoutMs, callback); }
  };
}

module.exports = {connect: connect};
