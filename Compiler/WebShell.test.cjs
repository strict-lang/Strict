const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const html = fs.readFileSync(process.argv[2] || `${__dirname}/WebShell.html`, 'utf8');
const elements = new Map();
const element = id => {
	if (!elements.has(id)) elements.set(id, { style: {}, addEventListener() {}, focus() { this.focused = true; } });
	return elements.get(id);
};
const context = { console, location: { pathname: '/Game.html' }, window: { addEventListener() {} },
	document: { getElementById: element, fullscreenEnabled: true }, requestAnimationFrame() {} };
vm.createContext(context);
vm.runInContext(html.match(/<script>([\s\S]*?)<\/script>/)[1], context);
assert.notEqual(context.Module.noInitialRun, true, 'Game must start without user interaction');
assert.doesNotMatch(html, /<button\b[^>]*\bid=["']?play\b/, 'Extra Play screen must be absent');
context.Module.onRuntimeInitialized();
assert.equal(element('overlay').style.display, 'none');
assert.equal(element('canvas').focused, true);
assert.equal(element('fullscreen').style.display, 'block');
context.Module.onAbort('missing asset');
assert.equal(element('overlay').style.display, 'flex');
assert.equal(element('status').textContent, 'Error: missing asset');
assert.equal(element('fullscreen').style.display, 'none');
console.log('Web startup and error overlay passed');
