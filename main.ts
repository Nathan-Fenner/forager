import { debug } from "util";
import { normalize } from "path";

const canvas = document.getElementById("canvas") as HTMLCanvasElement;
canvas.width = 600;
canvas.height = 600;

const debug_canvas = document.getElementById("debug_canvas") as HTMLCanvasElement;
debug_canvas.width = 600;
debug_canvas.height = 600;

const debug_ctx = debug_canvas.getContext("2d")!;

function clamp(low: number, value: number, high: number): number {
    return Math.max(low, Math.min(high, value));
}

class GridPos {
    static cache: {[k: string]: GridPos} = {};
    constructor(public readonly a: number, public readonly b: number, public readonly c: number) {
        if (a + b + c != 0) {
            throw {message: "invalid GridPos", args: [a, b, c]};
        }
        if (this.a == 0) {
            this.a = 0;
        }
        if (this.b == 0) {
            this.b = 0;
        }
        if (this.c == 0) {
            this.c = 0;
        }
        const k = [this.a, this.b, this.c].join(";");
        if (k in GridPos.cache) {
            return GridPos.cache[k];
        }
        GridPos.cache[k] = this;
    }
    shiftCoord(a: number, b: number, c: number): GridPos {
        if (a + b + c != 0) {
            throw {message: "shiftCoord", args: [a, b, c]};
        }
        return new GridPos(this.a + a, this.b + b, this.c + c);
    }
    shiftDirection(dir: number, amt: number = 1): GridPos {
        while (dir < 0) {
            dir = 6 + dir;
        }
        dir %= 6;
        const ex = origin.neighbors()[dir];
        return this.shiftCoord(ex.a * amt, ex.b * amt, ex.c * amt);
    }
    neighbors(): GridPos[] {
        return [
            this.shiftCoord(+1, +0, -1),
            this.shiftCoord(+0, +1, -1),
            this.shiftCoord(-1, +1, +0),
            this.shiftCoord(-1, +0, +1),
            this.shiftCoord(+0, -1, +1),
            this.shiftCoord(+1, -1, +0),
        ];
    }
    distance(p: GridPos): number {
        return (Math.abs(this.a - p.a) + Math.abs(this.b - p.b) + Math.abs(this.c - p.c)) / 2;
    }
    centerWorld(scale: number): [number, number] {
        const cx = this.a*scale*Math.sqrt(3)+this.b*scale*Math.sqrt(3)/2;
        const cy = this.b*scale*3/2;
        return [cx, cy];
    }
}

const origin = new GridPos(0, 0, 0);

type CellData = {
    height: number
}

const terra_map = new Map<GridPos, CellData>();
function addCell(cell: GridPos, data: CellData) {
    if (terra_map.has(cell)) {
        throw "cannot re-add cell";
    }
    terra_map.set(cell, data);
}

addCell(origin, {height: 3});
const frontier: {parent: GridPos, edge: GridPos}[] = [];
for (const neighbor of origin.neighbors()) {
    frontier.push({parent: origin, edge: neighbor});
}

while (terra_map.size < 300) {
    const heightDifference = () => Math.random() < 3/4 ? 0 : (Math.random() * 2 | 0)*2-1;
    const cellI = Math.random()*frontier.length | 0;
    const add = frontier[cellI];
    frontier[cellI] = frontier[frontier.length-1];
    frontier.pop();
    if (terra_map.has(add.edge)) {
        continue;
    }
    addCell(add.edge, {height: clamp(0, terra_map.get(add.parent)!.height + heightDifference(), 6)});
    for (const neighbor of add.edge.neighbors()) {
        frontier.push({parent: add.edge, edge: neighbor});
    }
}

// smooth

let changed = true;
let maxIter = 100;
while (changed && maxIter-- > 0) {
    changed = false;
    for (const [cell, data] of terra_map) {
        const adjacent = cell.neighbors().map(n => terra_map.get(n)).filter(x => x != undefined) as CellData[];
        if (adjacent.length > 0 && adjacent[0].height != data.height && adjacent.every((d: CellData): boolean => d.height == adjacent[0].height)) {
            changed = true;
            data.height = adjacent[0].height;
        }
    }
    for (const [cell, data] of terra_map) {
        const adjacent = cell.neighbors().map(n => terra_map.get(n)).filter(x => x != undefined) as CellData[];
        const heights = adjacent.map(x => x.height).filter(h => Math.abs(h - data.height) <= 1);
        if (range(heights) <= 1 && heights.filter(h => h == data.height).length <= 1) {
            const others = heights.filter(h => h != data.height);
            if (others.length > 0) {
                changed = true;
                data.height = others[0];
            }
        }
    }
}

function rgb(r: number, g: number, b: number): string {
    return `rgb(${clamp(0, 256*r|0, 255)},${clamp(0, 256*g|0, 255)},${clamp(0, 255*b|0, 255)})`;
}

const hex_size = 2;
debug_ctx.save();
debug_ctx.translate(300, 300);
for (const [p, data] of terra_map) {
    const r = data.height / 7;
    debug_ctx.fillStyle = rgb(r, r, r);
    debug_ctx.beginPath();
    const cx = p.a*hex_size*Math.sqrt(3)+p.b*hex_size*Math.sqrt(3)/2;
    const cy = p.b*hex_size*3/2;
    debug_ctx.moveTo(cx, cy);
    for (let i = 1; i < 6; i++) {
        debug_ctx.lineTo(cx+Math.cos(-Math.PI/2-i/3*Math.PI)*hex_size, cy+Math.sin(-Math.PI/2-i/3*Math.PI)*hex_size+hex_size);
    }
    debug_ctx.closePath();
    debug_ctx.fill();
}
debug_ctx.restore();

const gl = canvas.getContext("webgl2")! as WebGLRenderingContext;

const vertexPickSrc = `#version 300 es

precision mediump float;

uniform mat4 camera_perspective;
uniform mat4 camera_orientation;
uniform mat4 camera_position;

in vec3 a_pos;

in int a_obj;

out vec3 v_pos;
flat out int v_obj;

void main() {
    gl_Position = camera_perspective * camera_orientation * camera_position * vec4(a_pos, 1.0);
    v_pos = a_pos;
    v_obj = a_obj;
}

`;

const vertexShaderSrc = `#version 300 es

precision mediump float;

uniform mat4 camera_perspective;
uniform mat4 camera_orientation;
uniform mat4 camera_position;

in vec3 a_pos;
in vec3 a_color;
in vec3 a_normal;
in int a_obj;

out vec3 v_color;
out vec3 v_normal;
out vec3 v_pos;
flat out int v_obj; 

void main() {
    gl_Position = camera_perspective * camera_orientation * camera_position * vec4(a_pos, 1.0);
    v_color = a_color;
    v_normal = a_normal;
    v_pos = a_pos;
    v_obj = a_obj;
}

`;

const fragmentPickSrc = `#version 300 es

precision mediump float;

in vec3 v_pos;
flat in int v_obj;

out vec4 outColor;

void main() {
    int rInt = (v_obj / 400);
    int gInt = (v_obj / 20) % 20;
    int bInt = (v_obj) % 20;
    outColor = vec4(float(rInt) / 20.0, float(gInt) / 20.0, float(bInt) / 20.0, 1.0);
}

`;


const fragmentShaderSrc = `#version 300 es

precision mediump float;

in vec3 v_color;
in vec3 v_normal;
in vec3 v_pos;
flat in int v_obj;

uniform vec3 view_position;

uniform int u_selected;

float rand(vec2 c){
	return fract(sin(dot(c.xy ,vec2(12.9898,78.233))) * 43758.5453);
}
vec3 dir(vec2 p) {
    return vec3(rand(2. * p), rand(2. * p + 1.), rand(2. * p + vec2(1., 0.)));
}
vec3 val(vec2 c) {
    vec2 f = fract(c);
    vec2 b = c - f;
    vec3 n00 = dir(b);
    vec3 n01 = dir(b + vec2(0., 1.));
    vec3 n10 = dir(b + vec2(1., 0.));
    vec3 n11 = dir(b + vec2(1., 1.));

    vec3 n0 = mix(n00, n01, f.y);
    vec3 n1 = mix(n10, n11, f.y);
    return mix(n0, n1, f.x);
}
vec3 noise(vec2 p) {
    return val(p * 258.7) * 0.2 + val(p * 451.7) * 0.2 + val(p * 679.1) * 0.1;
}

out vec4 outColor;

void main() {
    vec3 light_direction = normalize(vec3(1.0, -4.0, 1.0));
    float lambert = max(0.1, dot(normalize(v_normal), light_direction));
    vec3 spec_normal = v_normal; // normalize( v_normal + 0.3 * noise(v_pos.xz) );
    float specular1 = pow(max(0.0, dot(reflect( normalize(view_position - v_pos), spec_normal ), -light_direction)), 10.0) * 0.125;
    float specular2 = pow(max(0.0, dot(reflect( normalize(view_position - v_pos), spec_normal ), -light_direction)), 100.0) * 0.25;

    vec3 albedo = v_color * exp( (2.0 - v_pos.y)*2.0 - 0.3 );

    if (u_selected == v_obj) {
        albedo = vec3(0.9, 0.8, 0.2);
        specular1 *= 3.0;
        specular2 *= 3.0;
    }
    outColor = vec4(lambert * albedo + specular1 + specular2, 1.0);
}

`;

const vertexPick = gl.createShader(gl.VERTEX_SHADER)!;
const fragmentPick = gl.createShader(gl.FRAGMENT_SHADER)!;

const vertexShader = gl.createShader(gl.VERTEX_SHADER)!;
const fragmentShader = gl.createShader(gl.FRAGMENT_SHADER)!;

gl.shaderSource(vertexPick, vertexPickSrc);
gl.shaderSource(fragmentPick, fragmentPickSrc);

gl.shaderSource(vertexShader, vertexShaderSrc);
gl.shaderSource(fragmentShader, fragmentShaderSrc);

gl.compileShader(vertexPick);

if (!gl.getShaderParameter(vertexPick, gl.COMPILE_STATUS)) {
    throw {message: "error loading vertex shader: " + gl.getShaderInfoLog(vertexPick)};
}

gl.compileShader(fragmentPick);

if (!gl.getShaderParameter(fragmentPick, gl.COMPILE_STATUS)) {
    throw {message: "error loading fragment shader: " + gl.getShaderInfoLog(fragmentPick)};
}

gl.compileShader(vertexShader);

if (!gl.getShaderParameter(vertexShader, gl.COMPILE_STATUS)) {
    throw {message: "error loading vertex shader: " + gl.getShaderInfoLog(vertexShader)};
}

gl.compileShader(fragmentShader);

if (!gl.getShaderParameter(fragmentShader, gl.COMPILE_STATUS)) {
    throw {message: "error loading fragment shader: " + gl.getShaderInfoLog(fragmentShader)};
}

const programPick = gl.createProgram();
gl.attachShader(programPick, vertexPick);
gl.attachShader(programPick, fragmentPick);
gl.linkProgram(programPick);

if (!gl.getProgramParameter(programPick, gl.LINK_STATUS)) {
    throw {message: "error linking programPick: " + gl.getProgramInfoLog(programPick)};
}


const program = gl.createProgram();
gl.attachShader(program, vertexShader);
gl.attachShader(program, fragmentShader);
gl.linkProgram(program);

if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
    throw {message: "error linking program: " + gl.getProgramInfoLog(program)};
}

const a_posPick = gl.getAttribLocation(programPick, "a_pos");
const a_objPick = gl.getAttribLocation(programPick, "a_obj");

const a_obj = gl.getAttribLocation(program, "a_obj");

const a_pos = gl.getAttribLocation(program, "a_pos");
const a_color = gl.getAttribLocation(program, "a_color");
const a_normal = gl.getAttribLocation(program, "a_normal");

const a_posPick_buffer = gl.createBuffer()!;
const a_objPick_buffer = gl.createBuffer()!;

const a_pos_buffer = gl.createBuffer()!;
const a_color_buffer = gl.createBuffer()!;
const a_normal_buffer = gl.createBuffer()!;

gl.enable(gl.DEPTH_TEST);


type Vec3 = [number, number, number];

function mag(v: Vec3): number {
    return (v[0]**2 + v[1]**2 + v[2]**2) ** 0.5;
}

function scale(k: number, v: Vec3): Vec3 {
    return [v[0] * k, v[1] * k, v[2] * k];
}
function unit(v: Vec3): Vec3 {
    return scale(1 / mag(v), v);
}
function cross(u: Vec3, v: Vec3): Vec3 {
    return [u[1]*v[2] - u[2]*v[1], u[2]*v[0] - u[0]*v[2], u[0]*v[1] - u[1]*v[0]];
}
function subtract(u: Vec3, v: Vec3): Vec3 {
    return [u[0] - v[0], u[1] - v[1], u[2] - v[2]];
}
function add(u: Vec3, v: Vec3): Vec3 {
    return [u[0] + v[0], u[1] + v[1], u[2] + v[2]];
}
function lerp(u: Vec3, t: number, v: Vec3): Vec3 {
    return add(scale(t, v), scale(1-t, u));
}
function withHeight(u: Vec3, h: number): Vec3 {
    return [u[0], h, u[2]];
}
function average(...xs: Vec3[]): Vec3 {
    const total: Vec3 = [0, 0, 0];
    for (const x of xs) {
        total[0] += x[0];
        total[1] += x[1];
        total[2] += x[2];
    }
    return scale(1 / xs.length, total);
}
function supplyY(xz: [number, number], y: number): Vec3 {
    return [xz[0], y, xz[1]];
}

const triangle_obj: number[] = [];

const triangle_pos: number[] = [];
const triangle_col: number[] = [];
const triangle_normal_bump: Vec3[] = [];
const triangle_normal: number[] = [];

function nub<T>(list: T[]): T[] {
    const out: T[] = [];
    for (const x of list) {
        if (out.indexOf(x) < 0) {
            out.push(x);
        }
    }
    return out;
}

function range(xs: number[]): number {
    return Math.max(...xs) - Math.min(...xs);
}

let obj = 0;

for (const [cell, data] of terra_map) {
    obj++;
    const colTop: Vec3 = [0.7, 0.7, 0.7]; // lerp([0.3, 0.3, 0.3], data.height/6, [0.9, 0.9, 0.9]);
    const colSide: Vec3 = [.5, .5, .5];

    const worldHeight = (h: number): number => 2 - h / 20;

    const size = 0.1;

    const ramp_depth = 0.3;

    const rim: ({type: "ramp", side: GridPos, p: Vec3} | {type: "corner", side1: GridPos, side2: GridPos, p: Vec3})[] = [];
    // rim is a circle.
    // it consists of corners, and optionally middles.
    // a corner with middles on both sides (with the same height) will be reduced to their height.
    // then, middles with two corners of the same height will be removed.
    // lastly, a middle with a corner of a different height will be split up.
    for (let c = 0; c < 6; c++) {
        const hex = supplyY(cell.centerWorld(size), worldHeight(data.height));
        const gi = cell.shiftDirection(c);
        const gj = cell.shiftDirection(c+1);
        const gk = cell.shiftDirection(c+2);
        const ci = supplyY(gi.centerWorld(size), worldHeight(data.height));
        const cj = supplyY(gj.centerWorld(size), worldHeight(data.height));
        const ck = supplyY(gk.centerWorld(size), worldHeight(data.height));
        const data_i = terra_map.get(gi) || data;
        const data_j = terra_map.get(gj) || data;
        const corner_ij = average(hex, ci, cj);
        const corner_jk = average(hex, cj, ck);
        const hs = [data.height, data_i.height, data_j.height];
        
        if (range(hs) <= 1) {
            rim.push({type: "corner", side1: gi, side2: gj, p: withHeight(corner_ij, worldHeight(Math.max(...hs)/2+Math.min(...hs)/2))});
        } else {
            rim.push({type: "corner", side1: gi, side2: gj, p: withHeight(corner_ij, worldHeight(data.height))});
        }
        if (Math.abs(data_j.height - data.height) <= 1) {
            rim.push({
                type: "ramp",
                side: gj,
                p: withHeight(lerp(corner_ij, 0.25, corner_jk), worldHeight(data.height/2 + data_j.height/2)),
            });
            rim.push({
                type: "ramp",
                side: gj,
                p: withHeight(lerp(corner_ij, 0.75, corner_jk), worldHeight(data.height/2 + data_j.height/2)),
            });
        }
    }
    for (let i = 0; i < rim.length; i++) {
        const iNext = (i+1)%rim.length;
        const iPrev = (i+rim.length-1)%rim.length;
        if (rim[i].type == "corner" && rim[iNext].type == "ramp" && rim[iPrev].type == "ramp" && rim[iPrev].p[1] == rim[iNext].p[1]) {
            rim[i].p = withHeight(rim[i].p, rim[iNext].p[1]);
        }
    }

    function addTriangle(a: Vec3, b: Vec3, c: Vec3, col: Vec3, options: {bump?: Vec3, obj: number}) {
        triangle_obj.push(options.obj, options.obj, options.obj);
        triangle_pos.push(...a, ...b, ...c);
        triangle_col.push(...col, ...col, ...col);
        triangle_normal_bump.push(options && options.bump || [0, 0, 0]);
    }

    function addQuad(a: Vec3, b: Vec3, c: Vec3, d: Vec3, col: Vec3, options: {bump?: Vec3, obj: number}) {
        if (Math.random() < 1/2) {
            addTriangle(a, b, d, col, options);
            addTriangle(d, b, c, col, options);
        } else {
            addTriangle(a, b, c, col, options);
            addTriangle(a, c, d, col, options);
        }
    }

    for (let i = 0; i < rim.length; i++) {
        let j = (i+1)%rim.length;
        const rim_i = rim[i];
        const rim_j = rim[j];
        const [hx, hy] = cell.centerWorld(size);
        const h: Vec3 = [hx, worldHeight(data.height), hy];

        const rails = [
            {posCenter: 0, heightCenter: 0},
            {posCenter: ramp_depth/3, heightCenter: 0},
            {posCenter: 2*ramp_depth/3, heightCenter: 1/2},
            {posCenter: ramp_depth, heightCenter: 1},
            {posCenter: 1, heightCenter: 1},
        ];
        const mix = (a: number, t: number, b: number): number => a*(1-t) + b*t;
        for (let r = 0; r < rails.length-1; r++) {
            if (rails[r+1].posCenter == 1) {
                addTriangle(
                    withHeight(lerp(rim_i.p, rails[r].posCenter, h), mix(rim_i.p[1], rails[r].heightCenter, h[1])),
                    withHeight(lerp(rim_j.p, rails[r].posCenter, h), mix(rim_j.p[1], rails[r].heightCenter, h[1])),
                    h,
                    colTop,
                    {obj},
                );
            } else {
                addQuad(
                    withHeight(lerp(rim_i.p, rails[r].posCenter, h), mix(rim_i.p[1], rails[r].heightCenter, h[1])),
                    withHeight(lerp(rim_j.p, rails[r].posCenter, h), mix(rim_j.p[1], rails[r].heightCenter, h[1])),
                    withHeight(lerp(rim_j.p, rails[r+1].posCenter, h), mix(rim_j.p[1], rails[r+1].heightCenter, h[1])),
                    withHeight(lerp(rim_i.p, rails[r+1].posCenter, h), mix(rim_i.p[1], rails[r+1].heightCenter, h[1])),
                    colTop,
                    {obj},
                );
            }
        }

        // Add the wall
        const lower = ([a, b, c]: Vec3): Vec3 => [a, 2, c];
        const sides: GridPos[] = [];
        sides.push(...(rim_i.type == "corner" ? [rim_i.side1, rim_i.side2] : [rim_i.side]));
        sides.push(...(rim_j.type == "corner" ? [rim_j.side1, rim_j.side2] : [rim_j.side]));
        if (sides.some(g => terra_map.has(g) ? terra_map.get(g)!.height <= terra_map.get(cell)!.height-1 : true)) {
            addQuad(rim_j.p, rim_i.p, lower(rim_i.p), lower(rim_j.p), colSide, {obj: 0});
        }
    }
    // add rocks to cliffs
    const rockColor: Vec3 = [0.8, 0.8, 0.8];
    for (let c = 0; c < 6; c++) {
        const n = cell.shiftDirection(c);
        if (!terra_map.has(n) || terra_map.get(n)!.height <= data.height-2) {
            // cliff
            const c = supplyY(cell.centerWorld(size), worldHeight(data.height));
            const mMiddle = lerp(supplyY(cell.centerWorld(size), worldHeight(data.height)), 0.5, supplyY(n.centerWorld(size), worldHeight(data.height)));
            
            const out = unit(withHeight(subtract(mMiddle, c), 0));
            const side: Vec3 = [-out[2], 0, out[0]];
            for (let i = 0; i < 6; i += 2) {
                const m = add(mMiddle, add( scale(Math.random() * 0.1 - 0.05, side), [0, Math.random() * -0.01 + i / 80, 0]   ))
                // TODO: do it better
                const rimOrtho: Vec3[] = [
                    [+0.01+i/400, -0.02, +0.02],
                    [+0.01+i/400, -0.02, -0.02],
                    [-0.01-i/400, -0.02, -0.02],
                    [-0.01-i/400, -0.02, +0.02],
                ];

                const rim: Vec3[] = rimOrtho.map(p => withHeight(add(scale(p[0], out), scale(p[2], side)), p[1]));
                
                addQuad(
                    add(rim[3], m),
                    add(rim[2], m),
                    add(rim[1], m),
                    add(rim[0], m),
                    rockColor,
                    {obj: 0},
                );
                for (let i = 0; i < rim.length; i++) {
                    let j = (i+1)%rim.length;
                    addQuad(
                        add(rim[i], m),
                        add(rim[j], m),
                        withHeight(add(rim[j], m), 2),
                        withHeight(add(rim[i], m), 2),
                        rockColor,
                        {obj: 0},
                    )
                }
            }
        }
    }
}

for (let t = 0; t < triangle_pos.length; t += 9) {
    const a: Vec3 = [triangle_pos[t+0], triangle_pos[t+1], triangle_pos[t+2]];
    const b: Vec3 = [triangle_pos[t+3], triangle_pos[t+4], triangle_pos[t+5]];
    const c: Vec3 = [triangle_pos[t+6], triangle_pos[t+7], triangle_pos[t+8]];
    const normal = add(unit(cross(subtract(b, a), subtract(c, a))), triangle_normal_bump[t/9]);
    triangle_normal.push(...normal);
    triangle_normal.push(...normal);
    triangle_normal.push(...normal);
}

gl.bindBuffer(gl.ARRAY_BUFFER, a_posPick_buffer);
gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(triangle_pos), gl.STATIC_DRAW);

gl.bindBuffer(gl.ARRAY_BUFFER, a_objPick_buffer);
gl.bufferData(gl.ARRAY_BUFFER, new Int32Array(triangle_obj), gl.STATIC_DRAW);

gl.bindBuffer(gl.ARRAY_BUFFER, a_pos_buffer);
gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(triangle_pos), gl.STATIC_DRAW);

gl.bindBuffer(gl.ARRAY_BUFFER, a_color_buffer);
gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(triangle_col), gl.STATIC_DRAW);

gl.bindBuffer(gl.ARRAY_BUFFER, a_normal_buffer);
gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(triangle_normal), gl.STATIC_DRAW);


function makePerspective(options: {zoom?: number}): number[] {
    const zoom = options.zoom || 1;
    const near = 0.1;
    const far = 100;
    return [
        zoom, 0, 0, 0,
        0, zoom, 0, 0,
        0, 0, (near+far) / (near-far), -1,
        0, 0, near*far/(near-far)*2, 0,
    ];
}

function makeCamera(options: {from: Vec3, forward: Vec3}): [number[], number[]] {
        // TODO: allow roll
        options.forward = unit(options.forward);
        const right: Vec3 = unit(cross(options.forward, [0, 1, 0]));
        const up: Vec3 = cross(options.forward, right);
        return [
            [
                right[0], up[0], options.forward[0], 0,
                right[1], up[1], options.forward[1], 0,
                right[2], up[2], options.forward[2], 0,
                0, 0, 0, 1,
            ],
            [
                1, 0, 0, 0,
                0, 1, 0, 0,
                0, 0, 1, 0,
                -options.from[0], -options.from[1], -options.from[2], 1,
            ],
        ];
}

const pickBuffer = gl.createFramebuffer()!;
if (pickBuffer == null) {
    throw "no frame buffer";
}
gl.bindFramebuffer(gl.FRAMEBUFFER, pickBuffer);

const pickDepthBuffer = gl.createRenderbuffer()!;
gl.bindRenderbuffer(gl.RENDERBUFFER, pickDepthBuffer); // create depth buffer
gl.renderbufferStorage(gl.RENDERBUFFER, gl.DEPTH_COMPONENT16, canvas.width, canvas.height); // allocate space
gl.framebufferRenderbuffer(gl.FRAMEBUFFER, gl.DEPTH_ATTACHMENT, gl.RENDERBUFFER, pickDepthBuffer);


const pickColor = gl.createTexture()!;
gl.bindTexture(gl.TEXTURE_2D, pickColor);
gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, canvas.width, canvas.height, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);

gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, pickColor, 0);
gl.bindTexture(gl.TEXTURE_2D, pickColor);

let doSelect: null | {x: number, y: number} = null;
let selectedObject: null | number = null;
let cameraCenter = {x: 0, z: 0};
let cameraForwardTarget = 0;
let cameraForward = 0;

const mouse = {
    x: 0,
    y: 0,
    dX: 0,
    dY: 0,
    down: {left: false, right: false},
};

canvas.onmousemove = e => {
    mouse.dX += e.offsetX - mouse.x;
    mouse.dY += e.offsetY - mouse.y;
    mouse.x = e.offsetX;
    mouse.y = e.offsetY;
};
canvas.onmousedown = function(e) {
    e.preventDefault();
    if (e.button == 0) {
        doSelect = {x: e.offsetX, y: canvas.height - e.offsetY};
        mouse.down.left = true;
    }
    if (e.button == 2) {
        mouse.down.right = true;
    }
    return false;
};
document.body.onmouseup = function(e) {
    if (e.button == 0) {
        mouse.down.left = false;
    }
    if (e.button == 2) {
        mouse.down.right = false;
    }
};
canvas.oncontextmenu = function(e) {
    e.preventDefault();
    return false;
};
document.body.onkeydown = function(e) {
    if (e.key == "d") {
        cameraForwardTarget += Math.PI / 3;
    }
    if (e.key == "a") {
        cameraForwardTarget -= Math.PI / 3;
    }
}

let zoom = 6;
let zoomTarget = 6;


document.body.onwheel = function(e) {
    zoomTarget *= Math.exp(e.wheelDeltaY / 1000);
    zoomTarget = Math.max(3, Math.min(30, zoomTarget));
}

function loop() {
    zoom = Math.exp(0.9 * Math.log(zoom) + 0.1 * Math.log(zoomTarget));
    const lookAngle = (90 - Math.log(zoom)*20) * Math.PI/180;
    const perspective = makePerspective({zoom});
    cameraForward = cameraForward * 0.9 + cameraForwardTarget * 0.1;
    if (mouse.down.right) {
        cameraCenter.x += (mouse.dX*Math.sin(cameraForward) - mouse.dY*Math.cos(cameraForward)/Math.sin(lookAngle)) * 0.022 / zoom;
        cameraCenter.z += (-mouse.dX*Math.cos(cameraForward) - mouse.dY*Math.sin(cameraForward)/Math.sin(lookAngle)) * 0.022 / zoom;
    }
    // tan(angle) = d
    const forward: Vec3 = unit([Math.cos(cameraForward), -Math.tan(lookAngle), Math.sin(cameraForward)]);
    // center + scale(t, forward) s.t. centerY + t*forward[1] == 0.5;
    const centerY = 1.75;
    const cameraY = -3;
    const from = add([cameraCenter.x, centerY, cameraCenter.z], scale( (cameraY - centerY) / forward[1] , forward));

    // const from: Vec3 = add([cameraCenter.x, 0.5, cameraCenter.z], scale(0.5, unit(withHeight(forward, 0))));
    
    const [cam_orient, cam_pos] = makeCamera({from, forward});

    // Picking
    gl.bindFramebuffer(gl.FRAMEBUFFER, pickBuffer);
    gl.useProgram(programPick);

    gl.bindBuffer(gl.ARRAY_BUFFER, a_posPick_buffer);
    gl.vertexAttribPointer(a_posPick, 3, gl.FLOAT, false, 0, 0);
    gl.enableVertexAttribArray(a_posPick);

    gl.bindBuffer(gl.ARRAY_BUFFER, a_objPick_buffer);
    (gl as any).vertexAttribIPointer(a_objPick, 1, gl.INT, false, 0, 0);
    gl.enableVertexAttribArray(a_objPick);

    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    {
        gl.uniformMatrix4fv(gl.getUniformLocation(programPick, "camera_perspective"), false, new Float32Array(perspective));
        gl.uniformMatrix4fv(gl.getUniformLocation(programPick, "camera_orientation"), false, new Float32Array(cam_orient));
        gl.uniformMatrix4fv(gl.getUniformLocation(programPick, "camera_position"), false, new Float32Array(cam_pos));
        gl.drawArrays(gl.TRIANGLES, 0, triangle_pos.length/3);
    }
    if (doSelect) {
        const pixelColors = new Uint8Array(canvas.width*canvas.height*4);
        gl.readPixels(0, 0, canvas.width, canvas.height, gl.RGBA, gl.UNSIGNED_BYTE, pixelColors);
        const i0 = pixelColors[(canvas.width*doSelect.y + doSelect.x)*4+0];
        const i1 = pixelColors[(canvas.width*doSelect.y + doSelect.x)*4+1];
        const i2 = pixelColors[(canvas.width*doSelect.y + doSelect.x)*4+2];
        const i = Math.floor( i0 / (255 / 20) + 0.5)*400 + Math.floor( i1 / (255/20) + 0.5)*20 + Math.floor(i2 / (255/20) + 0.5);
        selectedObject = i == 0 ? null : i;
        doSelect = null;
    }

    // Rendering
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.useProgram(program);

    gl.bindBuffer(gl.ARRAY_BUFFER, a_pos_buffer);
    gl.vertexAttribPointer(a_pos, 3, gl.FLOAT, false, 0, 0);
    gl.enableVertexAttribArray(a_pos);
    gl.bindBuffer(gl.ARRAY_BUFFER, a_color_buffer);
    gl.vertexAttribPointer(a_color, 3, gl.FLOAT, false, 0, 0);
    gl.enableVertexAttribArray(a_color);
    gl.bindBuffer(gl.ARRAY_BUFFER, a_normal_buffer);
    gl.vertexAttribPointer(a_normal, 3, gl.FLOAT, false, 0, 0);
    gl.enableVertexAttribArray(a_normal);

    gl.bindBuffer(gl.ARRAY_BUFFER, a_objPick_buffer);
    (gl as any).vertexAttribIPointer(a_obj, 1, gl.INT, false, 0, 0);
    gl.enableVertexAttribArray(a_obj);

    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    {
        gl.uniformMatrix4fv(gl.getUniformLocation(program, "camera_perspective"), false, new Float32Array(perspective));
        gl.uniformMatrix4fv(gl.getUniformLocation(program, "camera_orientation"), false, new Float32Array(cam_orient));
        gl.uniformMatrix4fv(gl.getUniformLocation(program, "camera_position"), false, new Float32Array(cam_pos));
        gl.uniform3f(gl.getUniformLocation(program, "view_position"), from[0], from[1], from[2]);
        gl.uniform1i(gl.getUniformLocation(program, "u_selected"), selectedObject ? selectedObject : -1);

        gl.drawArrays(gl.TRIANGLES, 0, triangle_pos.length/3);
    }

    mouse.dX = 0;
    mouse.dY = 0;
}
setInterval(loop, 10);
