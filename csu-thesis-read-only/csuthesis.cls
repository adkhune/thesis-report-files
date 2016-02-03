% csuthesis class
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Designed by Leif Anderson to meet the bullshit
% requirements of the Colorado State University
% Graduate School for the formatting of theses
% and dissertations.  This document should comply
% with the requirements as of spring 2014, by the
% time it is completed.
% Much additional input provided by Dr. Dan Brake
% and... others to come

% Licensed under the CC-GNU GPL version 2.0 or later.
% See http://creativecommons.org/licenses/GPL/2.0

% Areas that you might need to copy to your own header,
% or maybe comment out, are marked with big lines like the
% following:
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{csuthesis}[2014/06/02 Colorado State University Thesis class]


% This is basically going to be a class wrapper that takes
% amsbook and changes the formatting as little as possible
% to make it conform to  the dumber, uglier CSU standards.

% Class options:
\newif\ifphd \phdtrue
\newif\ifmasterofscience \masterofsciencetrue
\newif\ifsmallcapsok \smallcapsoktrue
\newif\ifcopyrightpage \copyrightpagefalse
\DeclareOption{masters}{\phdfalse}
\DeclareOption{nonscience}{\masterofsciencefalse}
\DeclareOption{masterof}{\phdfalse\masterofsciencefalse}
%there are a lot of different types of masters.
\DeclareOption{nosmallcaps}{\smallcapsokfalse}
% \DeclareOption{copyrightpage}{\copyrightpagetrue}
\DeclareOption{copyrightpage}{\OptionNotUsed}
% (As of feb 2014 the copyright page is mandatory.)
\DeclareOption{twoside}{\OptionNotUsed}
\DeclareOption*{\PassOptionsToClass{\CurrentOption}{amsbook}}
\ProcessOptions\relax
\LoadClass[12pt, oneside]{amsbook}

% User-specified global stuff (author, department, etc):
\makeatletter
	\global \let \@author \@empty 
	\global \let \@date \@empty 
	\global \let \@title \@empty
	\global \let \@departmentname \@empty
% 	\global \let \@committee \@empty
	\global \let \@gradterm \@empty
% 	\global \let \@gradyear \@empty
	\gdef\@gradyear{\the\year}
  \gdef\@masterof{Fine Arts}
	\global \let \@advisor \@empty
	
	\newcommand{\committee}[1]{\gdef\@committee{\xandlist{\\ \> }{\\ \>}{\\ \>}{#1}}}%
	\newcommand{\@committee}{\@latex@warning@no@line{No \noexpand\committee given}}%
	
	\def\author#1{\gdef\@author{#1}}%
	\def\@author{\@latex@warning@no@line{No \noexpand\author given}}
	\def\departmentname#1{\gdef\@departmentname{#1}}
	\def\gradterm#1{\gdef\@gradterm{#1}}
	\def\gradyear#1{\gdef\@gradyear{#1}}
  \def\masterof#1{\gdef\@masterof{#1}}
	\def\advisor#1{\gdef\@advisor{#1}}
	\def\coadvisor#1{\gdef\@coadvisor{#1}}
	\def\copyrighttext#1{\gdef\@copyrighttext{#1}}%no longer used
\makeatother

% margins
\usepackage[top=1in,bottom=1in,left=1in,right=1in]{geometry}
\footskip 0.5in %not fully tested, but I think this is the thing to modify.
% \usepackage[top=.8125in,bottom=.8125in,left=.8125in,right=.8125in]{geometry}

% Goddamn sideways shit:

% \usepackage{pdflscape}
\usepackage[pdftex]{lscape}
\usepackage{float}
\usepackage{afterpage}
\usepackage{environ}
\makeatletter
% \newenvironment{sidewayspage}{
% % 	store shit in a box
% \newsavebox{\tempboxside}
% \begin{lrbox}{\tempboxside}\begin{minipage}{\textheight}
% }{
% \end{minipage}\end{lrbox}
% % 	Write the box into the output
% \afterpage{
% 	\clearpage
% 	\begin{landscape}
% 	\thispagestyle{empty}
% % 	dump box here
% 	\usebox{\tempboxside}
% 	~\vfill\centerline{\footnotesize\thepage}
% 	\end{landscape}
% 	}
% }

% \newenvironment{sidewayspage}{
% % 	store shit in a box
% \newsavebox{\tempboxside}
% \begin{lrbox}{\tempboxside}
% }{
% \end{lrbox}
% % 	Write the box into the output
% \afterpage{
% 	\clearpage
% 	\begin{landscape}
% 	\thispagestyle{empty}
% % 	\begin{table}[!h]
% % 	dump box here
% 	\usebox{\tempboxside}
% % 	\end{table}
% 	~\vfill\centerline{\footnotesize\thepage}
% 	\end{landscape}
% 	}
% }

\NewEnviron{sidewayspage}{
% This is still a little hack-ish.  It runs into problems
% with sideways content that extends past one page.
% For multiple page sideways stuff, consider \begin{landscape}
% and a fancyhdr hack to move all pagenumbers sideways.
% The main goal here is to build a sideways endvironment that
% is sort of a float.
	\def\tempreacharound{\clearpage\begin{landscape}\thispagestyle{empty}~\vfill}
	\expandafter\expandafter\expandafter\afterpage\expandafter\expandafter\expandafter{\expandafter\tempreacharound % DON'T EVEN ASK
	% the \BODY command loses its meaning if it is used inside another environment... sometimes.  Trust me, we need to do this.
	% More to the point, the whole afterpage thing makes this landscape page a floating page.
	% the \clearpage empties out the other floats.  Then \thispagestyle clears the page number from the old bottom
	\BODY
	% consider making the sidewayspage environment more of a table wrapper, with \captionof{table}{.... caption ...\label{table:sideways}}
	~\vfill\centerline{\footnotesize\thepage}%this puts the page number back in, but over on the side.
	%there could be problems here if the user changes page number sizes or something.  This one will not change automatically.
	\end{landscape}}
}

% \NewEnviron{sidewayspage}{%
% \afterpage{\clearpage\begin{landscape}\thispagestyle{empty} \BODY ~\vfill\centerline{\footnotesize\thepage}\end{landscape}}
% }

% \newfloat{sidewayspage}{p}{aux}[table]
% \floatname{sidewayspage}{Table}
\makeatother

% spacing
% it's taken some tinkering to get the spacing to work out correctly.
% I think that some commands are designed to work in the body, or in
% the actual header, so that when included here, they fail.
\usepackage{setspace}
% \doublespace
% \setstretch{1.66}
% \setstretch{2}
% \usepackage{doublespace}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% OH MY GOD THE GRAD SCHOOL WAS ANGRY ABOUT THE SPACE 
% AFTER FIGURE 3.2!
% Copy some of these lines to the header (before 
% \begin{document}) of your own file, and change the numbers
% around until the big space goes away.
\setstretch{1.95} %line spacing.  
% Should be 2 for double spacing, but this works better.
\setlength{\intextsep}{12pt plus 2pt minus 2pt} 
%spacing above+below a figure/table that has text above and below it
\setlength{\textfloatsep}{12pt plus 2pt minus 2pt} 
%spacing below/above a figure or table at the top/bottom of a page.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% figures
\usepackage[pdftex]{graphicx}

% page headings and numbering
\makeatletter
% 	\usepackage{fancyhdr}
% ^ Not needed.  All we're doing is stripping away formatting.
% This works perfectly:
	\pagestyle{plain}
\makeatother


% \setcounter{tocdepth}{3}  %limits how deep sections are listed in the ToC, and how far they are numbered in the body.
   %  1 goes to sections, 2 to subsections, etc.

% See the ams manual thing, page 40 or so, for the beginning of the section, subsection, and table of contents formatting stuff.
\makeatletter
% \renewcommand{\contentsname}{\normalfont \normalsize \MakeUppercase{Table of Contents}}
% \renewcommand{\contentsname}{\MakeUppercase{Table of Contents}}
% \renewcommand{\listfigurename}{\MakeUppercase{List of Figures}}
% \renewcommand{\listtablename}{\MakeUppercase{List of Tables}}
% \renewcommand{\bibname}{\MakeUppercase{Bibliography}}
% \renewcommand{\indexname}{\MakeUppercase{Index}}
% Hack-ish:
% \ifsmallcapsok
% 	\renewcommand{\contentsname}{{\scshape Table of Contents}}
% 	\renewcommand{\listfigurename}{{\scshape List of Figures}}
% 	\renewcommand{\listtablename}{{\scshape List of Tables}}
% 	\renewcommand{\indexname}{{\scshape Index}}
% \else
% 	\renewcommand{\contentsname}{\MakeUppercase{Table of Contents}}
% 	\renewcommand{\listfigurename}{\MakeUppercase{List of Figures}}
% 	\renewcommand{\listtablename}{\MakeUppercase{List of Tables}}
% 	\renewcommand{\indexname}{\MakeUppercase{Index}}
% \fi
% ^ Handled better in another place now.
\renewcommand{\contentsname}{Table of Contents}
% \def\tableofcontents{%
%   \@starttoc{toc}\contentsname
% }
% \def\@makechapterhead#1{\global\topskip 7.5pc\relax %redefined not to have space.
\def\@makechapterhead#1{\global\topskip .625in\relax
  \begingroup
%   \fontsize{\@xivpt}{18}\bfseries\centering %from the rulebook: Do not use bold for preliminary page headings.
% I interpreted chapter headings as "preliminary page headings"
% addedthe scshape command here:
% If you want to change this, copy it to your document header,
% and modify your local copy.
	\normalsize\normalfont\scshape\centering
    \ifnum\c@secnumdepth>\m@ne
      \leavevmode \hskip-\leftskip
      \rlap{\vbox to\z@{\vss
          \centerline{\normalsize\mdseries
              \uppercase\@xp{\chaptername}\enspace\thechapter}
          \vskip 3pc}}\hskip\leftskip\fi
     \Large#1\par \endgroup%added the \Large to make the chapter headings larger.  The word "chapter" is left alone, only the name of the chapter gets bigger.
  \skip@34\p@ \advance\skip@-\normalbaselineskip
  \vskip\skip@ }
% \def\@makeschapterhead#1{\global\topskip 7.5pc\relax
\def\@makeschapterhead#1{\global\topskip .625in\relax
  \begingroup
%   \fontsize{\@xivpt}{18}\bfseries\centering %from the rulebook: Do not use bold for preliminary page headings.
% added scshape here:
  \normalsize\normalfont\scshape\centering
  #1\par \endgroup
  \skip@34\p@ \advance\skip@-\normalbaselineskip
  \vskip\skip@ }
\def\@makescsuhead#1{\global\topskip 0pc\relax
  \begingroup
  \normalsize\normalfont\centering% this would be the "correct" spot to add smallcaps formatting, wouldn't it?  
%   #1\par~\\ \endgroup%add/sutract space here, probably.
	\ifsmallcapsok
		\scshape #1\par \endgroup
	\else
		\MakeUppercase{#1}\par \endgroup
	\fi
  \skip@34\p@ \advance\skip@-\normalbaselineskip
  \vskip\skip@ }%this def is used just to make sure the table of contents name does not get bold faced or moved around
\def\@starttoc#1#2{%
  \begingroup
  \setTrue{#1}%
  \let\secdef\@gobbletwo \chapter
  \let\@secnumber\@empty % for \@tocwrite and \chaptermark
  \ifx\contentsname#2%
  \else \@tocwrite{chapter}{#2}\fi
  \typeout{#2}\@xp\chaptermark\@xp{#2}%
%   \@makeschapterhead{#2}\@afterheading %this writes the TABLE OF CONTENTS header, since this function is called with #2 as \contentsname.
	\@makescsuhead{#2}\@afterheading
  \parskip\z@skip
  \makeatletter
  \@input{\jobname.#1}%
  \if@filesw
    \@xp\newwrite\csname tf@#1\endcsname
    \immediate\@xp\openout\csname tf@#1\endcsname \jobname.#1\relax
  \fi
  \global\@nobreakfalse \endgroup
  \newpage
}
% \renewcommand{\section}{\@startsection
% {section}%                   % the name
% {1}%                         % the level
% {0mm}%                       % the indent
% {\baselineskip}%            % the before skip
% {0.5\baselineskip}%          % the after skip
% {\normalfont\centering}} % the style
% To modify these, best practice is to copy them into your header
% and modify your header.  That way your changes will be persistent
% and not overwritten if you download an updated version of the class.
\def\section{\@startsection{section}{1}%
  \z@{.7\linespacing\@plus\linespacing}{.5\linespacing}%
  {\normalfont\scshape\centering}} %writes the section headings in smallcaps, centered.
\def\subsection{\@startsection{subsection}{2}%
  \normalparindent{.5\linespacing\@plus.7\linespacing}{-.5em}%
  {\normalfont\scshape}}%writes the subsection headings in smallcaps.
\def\subsubsection{\@startsection{subsubsection}{3}%
  \normalparindent\z@{-.5em}%
  {\normalfont\itshape}}%sub sub sections are italics.


% Thanks to Chris Strickland for these two lines.
% They should prevent three-digit figure and table numbers
% from running over the name of the fig/tab in the T.o.C.
% \def\l@figure#1#2{\@tocline{0}{3pt plus2pt}{0pt}{2.3pc}{}{#1}{#2}}
% \def\l@table#1#2{\@tocline{0}{3pt plus2pt}{0pt}{2.3pc}{}{#1}{#2}}
% Actually, this might be more appropriate: 
% (copied from amsbook, modified)
\def\l@figure{\@tocline{0}{3pt plus2pt}{0pt}{2.3pc}{}}
\let\l@table=\l@figure

% This batch of commands makes the chapter number show up within several of the other numbers.
\numberwithin{table}{chapter}
\numberwithin{figure}{chapter}
\numberwithin{section}{chapter}
% numberwithin is really more of a taste thing.  It's an
% option that can be set within amsbook.  However, there
% was one complaint about table numbers not being 
% continuous without it (multiple table 1's, from the 
% multiple chapters).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% bibliography
% if you use chicago style citations, comment out
% everything between the %%%%%%%%%%% lines.  Then see the
% example file dealing with chicago citations for some
% hints about how to modify your own files.
\usepackage[square,comma,numbers,compress]{natbib} 
% better placed at the document level?
% Soon I will be gone, and it will be someone else's problem
% to wonder if the changes below are specific to natbib or if
% they are general enough to work for any bibliography.

% \renewcommand\bibsection{\chapter*{\bibname}}{}%existing command
% \renewcommand{\@bibtitlestyle}{%
%   \@xp\chapter\@xp*\@xp{\bibname}%
% }
% come back to this:
% \renewcommand\bibsection{\begin{singlespace} \begin{center} {\large \scshape \bibname} \addcontentsline{toc}{chapter}{{\scshape \bibname}}  \vspace{-0.5\baselineskip}\end{center}\end{singlespace}}
\renewcommand{\bibsection}{
	\ifsmallcapsok
		\centering {\large \scshape \bibname}
% 		\addcontentsline{toc}{chapter}{{\scshape \bibname}}
	\else
		\centering {\large \MakeUppercase{\bibname}}
% 		\addcontentsline{toc}{chapter}{\MakeUppercase{\bibname}}
	\fi
	\addcontentsline{toc}{chapter}{\bibname}}
% seems like this sort of overwrite is the better spot to
% make formatting changes, rather than at the \bibname level.  
\setlength{\bibsep}{0mm plus .5\baselineskip}
% This allows a little bit of stretch between bibliography
% items, so that the spacing at the top does not get
% stretched.  that used to mess up the formatting.

% We had two ~\\ up in there.
% uppercase was a name change, but smallcaps is a font change.
% This is still not quite perfect.  I feel like a perfect 
% solution would use chapter correctly (since the formatting
% rule for this is that it has to match chapter headings).

% End of the bibliography modifications section.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% \renewcommand{\listfigurename}{\normalfont \normalsize \MakeUppercase{List of Figures}}
% \renewcommand{\listtablename}{\normalfont \normalsize \MakeUppercase{List of Tables}}
% \renewcommand{\bibname}{\normalfont \normalsize BIBLIOGRAPHY}
% \renewcommand{\indexname}{\vspace{9mm plus 0mm minus 0mm} \hfill \centering \normalfont \normalsize INDEX \hfill}
% \thm@headfont{\normalfont}
% ^^ again, better handled elsewhere now.
\makeatother

% \makeatletter
% \newcommand\contentsnamefont{\upshape}
% \newcommand\contentsnamefont{\mdseries}
% \makeatother


% \usepackage{titletoc}

% DO NOT USE titletoc.
% It may be powerful, but the documentation is terrible
% and it requires too much work to understand all the 
% different arguments that get passed to the commands.

% \titlecontents{section} % set formatting for \section -
% 	                    % \subsection must be formatted separately 
% [2.3em]                 % adjust left margin
% {\rmfamily}             % font formatting
% {\contentslabel{2.3em}} % section label and offset
% {\hspace*{-2.3em}}
% {\titlerule*[1pc]{.}\contentspage}

% Dan's TOC solution:  (doesn't interact well with amsbook)
% \usepackage{tocloft}  %import the package  (Table of Contents)
% \renewcommand{\cftchapfont}{\normalfont}    %make chapter listings in ToC have normal font (not bold)
% \renewcommand\cftchappagefont{\normalfont}   %make chapter listings in ToC have normal font (not bold)
% \renewcommand{\cftchapleader}{\cftdotfill{\cftsecdotsep}}  %make sure chapters get dots, as per graduate school
% \renewcommand{\cftsecfont}{\normalfont}     
% \renewcommand{\cftfigfont}{Figure }
% \renewcommand{\cfttabfont}{Table }  %leave the trailing space
% 
% \cftsecindent 2em %indentations in em's.  the size of an em changes depending on the font you choose
% \cftsubsecindent 4em
% 
% \renewcommand{\cftpartdotsep}{\cftdotsep}  %force dots  between parts, chapters, & page numbers
% \renewcommand{\cftchapdotsep}{\cftdotsep}
% 
% 
% \renewcommand{\contentsname}{\vspace{-26.5mm plus 0mm minus 0mm} \hfill\normalsize \normalfont \centering TABLE OF CONTENTS \hfill}
% \renewcommand\listfigurename{\vspace{-27.25mm plus 0mm minus 0mm} \hfill \centering \normalfont \normalsize LIST OF FIGURES \hfill}
% \renewcommand{\bibname}{\vspace{3.25mm plus 0mm minus 0mm} \hfill \centering \normalfont \normalsize BIBLIOGRAPHY \hfill}
% \renewcommand{\indexname}{\vspace{9mm plus 0mm minus 0mm} \hfill \centering \normalfont \normalsize INDEX \hfill}
% <end of Dan's solution>

% Here is my solution, which is an overwrite of the more basic definition.

%dotted line for TOC.  This is the definition of \@tocline, copied out of amsbook.
\makeatletter
\def\@tocline#1#2#3#4#5#6#7{\relax
  \ifnum #1>\c@tocdepth % then omit
  \else
    \par \addpenalty\@secpenalty\addvspace{#2}%
    \begingroup \hyphenpenalty\@M
    \@ifempty{#4}{%
      \@tempdima\csname r@tocindent\number#1\endcsname\relax
    }{%
      \@tempdima#4\relax
    }%
    \parindent\z@ \leftskip#3\relax \advance\leftskip\@tempdima\relax
    \rightskip\@pnumwidth plus4em \parfillskip-\@pnumwidth
    #5\leavevmode\hskip-\@tempdima #6\nobreak\relax
%     \hfil\hbox to\@pnumwidth{\@tocpagenum{#7}}\par % this was the original line.
    \dotfill\hbox to\@pnumwidth{\@tocpagenum{#7}}\par % the dotfill makes a dotted line across to the page number.
    \nobreak
    \endgroup
  \fi}
\makeatother

%%%%%%%%%%%%%
%   title   %
%%%%%%%%%%%%%

%required stuff:
% 1 inch margins all sides, no bold, no titles on committee
% centered:
% Top line: 
% DISSERTATION (all caps)
% some space
% TITLE IN ALL CAPS AND DOUBLESPACED \MakeUppercase{} will be useful.
% some space
% Submitted by
% blank line
% \@author
% blank line
% \@departmentname
% some space
% In partial fulfullment of the requirements 
% blank line
% For the degree of Doctor of Philosophy
% blank line
% Colorado State University
% blank line
% Fort Collins, Colorado
% blank line
% \@gradterm
% end center
% left align:
% Doctoral Committee:
% blank line
% Indent, still left:
% Advisor: \@advisor
% Co-Advisor: \@coadvisor (if the coadvisor exists)
% blank line
% committee member list (no titles attached anywhere)
% NO PAGE NUMBER ON THIS PAGE but it is page 1 of frontmatter (so page i, unlabeled)
% That last part is a little silly, since it is immediately followed
% by an unlabeled AND uncounted copyright page.  Better to just 
% specify that the abstract is page ii.

% As a leadin to the title page, here is the Abstract 
% definition, copied from AMSbook and modified:
\makeatletter
\renewenvironment{abstract}{%
	\ifx\maketitle\relax
		\ClassWarning{\@classname}{Abstract should precede
			\protect\maketitle\space in CSU Thesis document class; reported}%
	\fi
	\global\setbox\abstractbox=\vtop \bgroup
% 		\normalfont\Small
% 		\begin{center}\scshape
% 		\abstractname \\
% 		\vspace{.5\baselineskip}
% 		\@title
% 		\end{center}
		\list{}{\labelwidth\z@
			\leftmargin0pc \rightmargin\leftmargin
			\listparindent\normalparindent \itemindent\z@
			\parsep\z@
			\itemsep0.5\baselineskip
			% In order to get equation numbers indented with the rest of the abstract, we have to do this:
			\let\fullwidthdisplay\relax
		}%
	\ifsmallcapsok
		\item \begin{center}\scshape Abstract\end{center}
		\item \begin{center}\scshape \@title\end{center}
% 		\addcontentsline{toc}{chapter}{{\scshape Abstract}}
	\else
		\item \begin{center}\MakeUppercase{Abstract}\end{center}
		\item \begin{center}\MakeUppercase{\@title}\end{center}
% 		\addcontentsline{toc}{chapter}{\MakeUppercase{Abstract}}
	\fi
	\addcontentsline{toc}{chapter}{Abstract}
	\item[\hskip\normalparindent]
% % 	\begin{center}\begin{singlespace}\MakeUppercase{Abstract}\\ ~\\\end{singlespace}
% % 	\centerline{\scshape Abstract}
% % % 	~\\
% % 	\begin{center}\scshape \@title\end{center}
% 	\ifsmallcapsok
% 		\begin{center}\begin{singlespace}{\scshape Abstract}\\ ~\\\end{singlespace}
% 		{\scshape \@title}
% 		\addcontentsline{toc}{chapter}{{\scshape Abstract}}
% 	\else
% 		\begin{center}\begin{singlespace}{\MakeUppercase{Abstract}}\\ ~\\\end{singlespace}
% 		\MakeUppercase{\@title}
% 		\addcontentsline{toc}{chapter}{\MakeUppercase{Abstract}}
% 	\fi
% % 	\hspace{-\baselineskip}
% 	\end{center}
% % 	\begin{singlespace}\vspace{\baselineskip}\end{singlespace}
% % 	\vspace{.5\baselineskip}
% 	\vskip 0pt plus 0pt minus 0pt
% 	%blank line below here so the abstract starts indented.
% 	
}{%
\endlist\egroup
% If the abstract was supposed to be typeset earlier, then \@setabstract is now equal to \relax, and we had better drop the contents of the abstract box onto the page immediately, to salvage the situation as best we can.
\ifx\@setabstract\relax \@setabstracta \fi
}
% AMS comment about abstract:
% % Because the abstract might be postponed until the end of an article, we can-not simply use the fact of a preceding \maketitle to tell whether \endabstract should immediately put the abstract box contents on the page. So we use an auxiliary function that will be reset to no-op once we have passed the point where the abstract should normally be typeset.
% This part should still be intact in the amsbook class.
% \def\@setabstract{\@setabstracta \global\let\@setabstract\relax}
% \def\@setabstracta{%
% \ifvoid\abstractbox
% \else
% 	\skip@20\p@ \advance\skip@-\lastskip
% 	\advance\skip@-\baselineskip \vskip\skip@
% 	\box\abstractbox
% 	\prevdepth\z@ % because \abstractbox is a vtop
% \fi
% }
% the acknowledgements will basically be a copy of the abstract.
% In fact, I copied the AMS definition of abstract, and changed a
% couple tiny things.  I even copied their comments mostly.
\newbox\ackbox
\newenvironment{acknowledgements}{%
  \ifx\maketitle\relax
    \ClassWarning{\@classname}{Acknowledgements should precede
      \protect\maketitle\space in the csuthesis class; reported}%
  \fi
  \global\setbox\ackbox=\vtop \bgroup
	\list{}{\labelwidth\z@
			\leftmargin0pc \rightmargin\leftmargin
			\listparindent\normalparindent \itemindent\z@
			\parsep\z@
			\itemsep0.5\baselineskip
			% In order to get equation numbers indented with the rest of the abstract, we have to do this:
			\let\fullwidthdisplay\relax
		}%
	\ifsmallcapsok
		\item \begin{center}\scshape Acknowledgements\end{center}
% 		\addcontentsline{toc}{chapter}{{\scshape Acknowledgements}}
	\else
		\item \begin{center}\MakeUppercase{Acknowledgements}\end{center}
% 		\addcontentsline{toc}{chapter}{\MakeUppercase{Acknowledgements}}
	\fi
	\addcontentsline{toc}{chapter}{Acknowledgements}
	\item[\hskip\normalparindent]
% 	\ifsmallcapsok
% % 		\begin{center}\scshape Acknowledgements\end{center}
% 		\centerline{\scshape Acknowledgements}
% 		\addcontentsline{toc}{chapter}{{\scshape Acknowledgements}}
% 	\else
% % 		\begin{center}{\MakeUppercase{Acknowledgements}}\end{center}%
% 		\centerline{\MakeUppercase{Acknowledgements}}
% 		\addcontentsline{toc}{chapter}{\MakeUppercase{Acknowledgements}}
% 	\fi
% 	\hspace{\baselineskip}
	%blank line below here so acknowledgements starts indented (to match abstract)
	
}{%
	% Old version of the 'easter egg':
	% 	Great thanks are also due to Leif Anderson for building this \LaTeX\ document class, allowing me to meet the graduate school formatting requirements with no effort on my part. % You're welcome.
	
	This \ifphd dissertation \else thesis \fi is typset in \LaTeX\ using a document class designed by Leif Anderson.%The more professional version.
	% Go ahead, comment it out.  But if you're here you probably know how much work went into this.
	% I call this my little "no need to thank me" easter egg.
  \endlist\egroup
  \ifx\@setack\relax \@setacka \fi
}
\def\@setack{\@setacka \global\let\@setack\relax}
\def\@setacka{%
  \ifvoid\ackbox
  \else
    \skip@20\p@ \advance\skip@-\lastskip
    \advance\skip@-\baselineskip \vskip\skip@
    \box\ackbox
    \prevdepth\z@ % because \abstractbox is a vtop (I have no idea what that means.)
  \fi
}
% Now we overwrite \maketitle
% the original had a lot more error handling, which I have ignored here.
% The original was also a lot more elegant.  Begin hack and slash.
% Our \maketitle will be huge, and will do everything needed at the beginning: title page, abstract, acknowledgements, copyright, etc.
\renewcommand{\maketitle}{%
	\begin{titlepage}
		\begin{center}
			\ifphd
				\ifsmallcapsok
					{\scshape Dissertation}
				\else
					\MakeUppercase{Dissertation}
				\fi
			\else
				\ifsmallcapsok
					{\scshape Thesis}
				\else
					\MakeUppercase{Thesis}
				\fi
			\fi
% 			{\large\scshape Dissertation}
% 			\textsc{Dissertation}
			~\vfill
% 			\begin{doublespace}
% 			{\MakeUppercase{\@title}}
% 			{\large\scshape \@title}
			\ifsmallcapsok
				{\scshape \@title}
			\else
				\MakeUppercase{\@title}
			\fi
% 			\end{doublespace}
% 			I think subtitles are not allowed.
% 			\ifx \@subtitle \@empty
% 				\relax
% 			\else
% 				\vspace{0.5em}
% 				{\large \@subtitle} \\
% 			\fi
			~\vfill
			Submitted by \\
% 			~\\
% 			blank line
			\ifx @author \@empty
				\relax% Look, error handling!
			\else
				\@author \\
			\fi
% 			\@author
% % 			blank line
% 			~\\
			\ifx\@empty\@departmentname
% 				\MakeUppercase{Department Name not defined}
				\emph{Attention!} $\backslash$departmentname \emph{not defined!}
			\else
				\@departmentname
			\fi
% 			\@departmentname
% % 			some space
			~\vfill
			In partial fulfillment of the requirements \\
			\ifphd
				For the Degree of Doctor of Philosophy \\ % The inconsistent capitalization between this line and the previous is required by the graduate school.
			\else
				\ifmasterofscience
					For the Degree of Master of Science \\
				\else
					For the Degree of Master of \@masterof \\
				\fi
			\fi
			Colorado State University \\
			Fort Collins, Colorado \\
			\ifx\@gradterm\@empty
% 				\MakeUppercase{gradterm not defined}
				\emph{Attention!} $\backslash$gradterm \emph{not defined!}
			\else
				\@gradterm~\@gradyear
			\fi
			~\vfill
		\end{center}
		% left align:
		\begin{singlespace}
		\ifphd
			Doctoral Committee:\\
		\else
			Master's Committee:\\
		\fi
		% blank line
		\begin{tabbing}
		\hspace*{0.5in}\=
		% Indent, still left:
		Advisor: \@advisor \\ \>
		% Co-Advisor: \@coadvisor (if the coadvisor exists)
		\ifx\@coadvisor\undefined
			\relax
		\else
% 			\hspace*{0.5in}Co-Advisor: \@coadvisor \\
			Co-Advisor: \@coadvisor \\
		\fi
		% blank line
		~\\ \> %the extra \> moves back to the right tab stop, which the ~\\ will have dropped.
		% committee member list (no titles attached anywhere) \@committee contains commands that mesh with this tabbing.
		\@committee
		\end{tabbing}
		\end{singlespace}
	\end{titlepage}
% 	Old version:
% 	\ifx\@copyrighttext\undefined
%  		% If no copyright text is defined, do nothing.
% 		\relax
% 	\else
% 	Newer version, but still outdated:
% 	\ifcopyrightpage
		% Make a copyright page
% 		This is now automated a little better, as a document option.
% 		Copyright page is no longer optional as of feb 2014.
% 		Now the option is removed and it just always happens.
		\thispagestyle{empty}
		\vspace*{\fill}
		\begin{center}
% 		\@copyrighttext
% 		Copyright by \@author\, \the\year \\ All Rights Reserved
		Copyright by \@author~\@gradyear \\ All Rights Reserved
% 		NOTE that this defaults to the current year.
% 		If you want to copyright in a different year, change this.
		\end{center}
		\vspace*{\fill}
		\clearpage
% 	\fi
	\setcounter{page}{2}
	% the copyright page is not assigned a number nor counted
	% thus the abstract always starts on page ii.
	\unvbox\abstractbox
% 	no error handling, because the grad school requires an abstract.
% 	\addcontentsline{toc}{chapter}{\MakeUppercase{Abstract}}
% 	contents line is added in abstract definition.
	% Acknowledgements:
	\ifvoid\ackbox
% 	if no acknowledgements page, do nothing. (Maybe add a \relax?  Seems to work without it.)
	\else
		\clearpage
		\unvbox\ackbox
% 		\addcontentsline{toc}{chapter}{\MakeUppercase{Acknowledgements}}
% 		add to contents command is in ack def.
	\fi
}


\newcommand{\preliminarypage}[2]{
	\clearpage
	\list{}{\labelwidth\z@
			\leftmargin0pc \rightmargin\leftmargin
			\listparindent\normalparindent \itemindent\z@
			\parsep\z@
			\itemsep0.5\baselineskip
			\let\fullwidthdisplay\relax
		}%
	\ifsmallcapsok
		\item \begin{center}\scshape #1\end{center}
	\else
		\item \begin{center}\MakeUppercase{#1}\end{center}
	\fi
	\addcontentsline{toc}{chapter}{#1}
	\item[\hskip0pt]#2%this will produce no indentation.  Deal with it.
% 	use \indent to indent your first paragraph if you want to.
	\clearpage}

% \newcommand{\preliminarypage}[2]{
% 	\clearpage
% 	\ifsmallcapsok
% 		\begin{center}\scshape #1\end{center}
% % 		\addcontentsline{toc}{chapter}{{\scshape Acknowledgements}}
% 	\else
% 		\begin{center}\MakeUppercase{#1}\end{center}
% % 		\addcontentsline{toc}{chapter}{\MakeUppercase{Acknowledgements}}
% 	\fi
% % 	\vspace{.5\baselineskip}
% 	\addcontentsline{toc}{chapter}{#1}
% #2
% \clearpage}

\makeatother


% \makeatletter
% \usepackage[compact]{titlesec}
% Don't use titlesec.
